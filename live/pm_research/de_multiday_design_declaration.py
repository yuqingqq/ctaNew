"""THE DESIGN DECLARATION for the ruled multi-day Gate-1 run (R-547).

NO DATA IS TOUCHED BY THIS MODULE. It emits a declaration and proves, on
synthetic fixtures, that the declared rule can FAIL an arm that should
fail, PASS an arm that should pass, and REFUSE a day whose reference book
does not match its pinned digest. The run itself is a later, separate act
that the reviewer files on first (R-547 item 6).

WHAT THE USER RULED (R-547(A), verbatim in the register): Gate 1's control
is the REPLAY NULL -- random decisions, same count and same side split as
the arm, drawn from the arm's own decision population at the arm's own
theta, replayed through the SAME stateful cascade. Five named days decide
the section-7 stopping rule. Design and null are committed BEFORE data.

    python3 live/pm_research/de_multiday_design_declaration.py --selftest
    python3 live/pm_research/de_multiday_design_declaration.py --emit --output PATH
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import de_data_root as DR  # noqa: E402


#: THE VERSION LIVES IN ONE PLACE. It travelled in THREE and they
#: disagreed: v7 on disk carried `protocol ..._V4`, and its
#: `supersedes.path` named **v2** -- so a reader resolving the receipt
#: field (rule 13's whole premise: automated readers resolve fields, not
#: sidecars) walked from v7 straight past v3, v4, v5 and v6. The
#: filename, the protocol suffix and the head of the chain are now
#: DERIVED from this integer and a battery check asserts all three
#: agree.
VERSION = 8
PROTOCOL = f"P003_DE_MULTIDAY_GATE1_DESIGN_DECLARATION_V{VERSION}"
EXPECTED_CHECKS = 65

V1_DECLARATION = ("p003_de_multiday_gate1_design__20260906T031853Z.json",
                  "89ac8b15b83c91971c2e2a5b472cd0d6f32a4ba4659b42233afdd1"
                  "b781c7bd6f")
V2_DECLARATION = ("p003_de_multiday_gate1_design_v2__20260906T035617Z.json",
                  "c4da696f60ca62700d18551f523e571aab09bc5b8b42f6970112ab"
                  "a6490ce903")
V3_DECLARATION = ("p003_de_multiday_gate1_design_v3__20260906T040539Z.json",
                  "a1016a8762fffdfeb368c5ff217e8bcc83141b1567e218a2f1b69c"
                  "658a65289c")
V4_DECLARATION = ("p003_de_multiday_gate1_design_v4__20260906T042458Z.json",
                  "24db4e1bd5bfc04bbdcd13ecdd28bb4260d2854539ac90b7df40e6"
                  "71eee78a3d")
V5_DECLARATION = ("p003_de_multiday_gate1_design_v5__20260906T043134Z.json",
                  "dfc599ba46a7c4a7dd4d0e274df128a780f27c7ccc732e84a72df8"
                  "fed9046cec")
V6_DECLARATION = ("p003_de_multiday_gate1_design_v6__20260906T043936Z.json",
                  "966ca76d2803fa5aa45cb5b15c3b6eff498c7888e9d9bb33b53612"
                  "effcbd39d2")
V7_DECLARATION = ("p003_de_multiday_gate1_design_v7__20260906T045059Z.json",
                  "bd33daf5beb4774b40cadcf7239f830ee0ea3edd12b617b6d5a161"
                  "1f9213072f")
#: OLDEST FIRST. `supersedes.path` is the LAST element, never a typed
#: constant -- that is how v7 came to name v2.
DECLARATION_CHAIN = (V1_DECLARATION, V2_DECLARATION, V3_DECLARATION,
                    V4_DECLARATION, V5_DECLARATION, V6_DECLARATION,
                    V7_DECLARATION)

#: (1) R2's FLOOR, CALIBRATED -- measured on the consumed 08-24 hour, the
#: one population already seen, exactly as R4's 0.25 was set against
#: HAZARD's observed 0.4055.
HEAD_OVERLAP_OBSERVED = 1.0
HEAD_OVERLAP_FLOOR = 0.90
HEAD_OVERLAP_MEASUREMENT = {
    "cache": "de_section81_cache_v2_12.pkl",
    "asm_key_shape": "asm['by_arm'][(coin, head)][0]",
    "scored_CONDVALUE_X_SKEW_q1_arrival_composed_lgbm": 29813,
    "scored_HAZARD_OVER_SKEWED_REF_incumbent_linear_d": 29813,
    "intersection": 29813,
    "overlap_over_the_smaller": 1.0,
    "jaccard": 1.0,
}

#: (4) DEGENERACY BARS, declared NOW so nobody decides after seeing a day.
MIN_DECISIONS_PER_ARM_DAY = 30
SD_FLOOR_FRACTION = 0.25          # refuse when sd < f * |mean| of the null

#: (4) THE DECLARED LEDGER ROOT. The derivation resolves through the R-397
#: symlink and compares; a different root REFUSES with the root named.
DECLARED_LEDGER_ROOT = "/home/yuqing/ctaNew/data"

#: R-555's ruled set. Named here so the R7 assertion can compare the
#: LEDGER AS READ against the RULE, instead of against a count that was
#: true on the day it was typed.
RULED_DAY_SET = ("2026-09-03", "2026-09-04", "2026-09-05",
                 "2026-09-06", "2026-09-07", "2026-09-08")

#: (7) the two candidate day sets, DERIVED from the ledger, not listed.
LEDGER_CONJUNCTS = ("day_closed_calendar", "post_freeze_pass",
                    "era_pure", "day_quality_pass")
#: (1) THE TRUE STATE PER DAY, AS A FIELD. v1 and v2 carried the sentence
#: "the race scored a different object on these days, sealed and unread".
#: R-549(A) withdrew it: 09-01 and 09-02 were OPENED under the interim read
#: and are CONSUMED (RESULTS.md:681). The substantive half survives -- the
#: thetas were fixed on the consumed 08-24 hour and nothing about the ARMS
#: was chosen on any of these days -- and it is stated separately from the
#: withdrawn half so the two cannot travel together again.
OPENED_NONE = "none"
OPENED_INTERIM = "interim_read_of_frozen_candidate"
OPENED_DEV = "development_read"

DAY_READ_STATE = {
    "2026-08-29": {"previously_opened_for": OPENED_DEV,
                   "authority": "R-500 withdrew it from the race; R-502 "
                                "ratified ONE development read",
                   "what_was_read": "the FROZEN CANDIDATE, not the arms"},
    "2026-09-01": {"previously_opened_for": OPENED_INTERIM,
                   "authority": "R-549(A); RESULTS.md:681 records it as "
                                "CONSUMED",
                   "what_was_read": "the FROZEN CANDIDATE, not the arms"},
    "2026-09-02": {"previously_opened_for": OPENED_INTERIM,
                   "authority": "R-549(A); RESULTS.md:681 records it as "
                                "CONSUMED",
                   "what_was_read": "the FROZEN CANDIDATE, not the arms"},
    "2026-09-03": {"previously_opened_for": OPENED_NONE, "authority": None,
                   "what_was_read": None},
    "2026-09-04": {"previously_opened_for": OPENED_NONE, "authority": None,
                   "what_was_read": None},
    "2026-09-05": {"previously_opened_for": OPENED_NONE, "authority": None,
                   "what_was_read": None},
    # R-555's set reaches beyond the days that exist at declaration time.
    # A day that has not CLOSED cannot have been read, and that is a fact
    # about the calendar rather than a claim about anyone's discipline --
    # but it is recorded rather than left to the absent-key refusal, so
    # the runner's check has something to compare against.
    "2026-09-06": {"previously_opened_for": OPENED_NONE,
                   "authority": "R-555; not closed at declaration time",
                   "what_was_read": None},
    "2026-09-07": {"previously_opened_for": OPENED_NONE,
                   "authority": "R-555; not closed at declaration time",
                   "what_was_read": None},
    "2026-09-08": {"previously_opened_for": OPENED_NONE,
                   "authority": "R-555; not closed at declaration time",
                   "what_was_read": None},
}

DAYS = ("2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05")
ARMS = ("CONDVALUE_X_SKEW", "HAZARD_OVER_SKEWED_REF")
G = len(DAYS)
MULTIPLICITY = 2
MIN_DRAWS = 500
ALPHA = 0.05

#: Fixed on the CONSUMED 2026-08-24 13:50-14:50Z hour and pinned in two
#: places that must agree: BE's null artifact (`cells[arm].arm_filed.theta`)
#: and DE's arms emission (`arms[arm].identity`). Neither is re-fitted on
#: any of the five days -- that is what makes the days unconsumed.
THETA = {"CONDVALUE_X_SKEW": 0.32450609461933483,
         "HAZARD_OVER_SKEWED_REF": 0.43525926488298716}
THETA_PINS = {
    "CONDVALUE_X_SKEW": {
        "artifact": "data/pm_5min/derived/be_cancel_axis_null_v1.json",
        "sha256": "6951f57d2b8a23bd2d51f24d25659ddffc671c5932aaba89246b025"
                  "182c2fa08",
        "json_path": "cells.CONDVALUE_X_SKEW.arm_filed.theta",
        "head": "q1_arrival_composed_lgbm",
        "model_artifacts": {"lgbm_haz_btc.txt": "ec52055214a01ed5",
                            "lgbm_thresholds_btc.json": "0fa2f1f7a5a4c58f"}},
    "HAZARD_OVER_SKEWED_REF": {
        "artifact": "data/pm_5min/derived/be_cancel_axis_null_v1.json",
        "sha256": "6951f57d2b8a23bd2d51f24d25659ddffc671c5932aaba89246b025"
                  "182c2fa08",
        "json_path": "cells.HAZARD_OVER_SKEWED_REF.arm_filed.theta",
        "head": "incumbent_linear_d",
        "model_artifacts": {"linear_d_btc.json": "18701008c2bd18c6"}},
}

CASCADE_MACHINERY = {
    "path": "live/pm_research/be_cancel_axis_null.py",
    "owner": "BE",
    "de_does_not_reimplement_it": True,
    "why": "two implementations of one cascade is two cascades; the null "
           "must run through the SAME stateful policy the arm ran through "
           "or it is not a control for it (R-547 item 1)",
}


#: R-572(B)(4) -- THE SEED CONVENTION, WHICH LIVED ONLY IN `seed_for()`.
#: The literal `P003_GATE1_MULTIDAY`, the `|` separator, the field ORDER and
#: the **8-hex truncation** were implementation details of one function. Change
#: any one of them and every draw sequence in the run changes SILENTLY -- the
#: artifact would still say "seeded from the book digest" and still be
#: reproducible-looking, and no reader could tell the sequence had moved. They
#: are declared fields now, and `seed_from_convention()` below is a REFERENCE
#: implementation built from these fields alone; the battery drives it against
#: the runner's own `seed_for()` and, field by field, against mutated copies
#: that must DISAGREE.
SEED_CONVENTION = {
    "hash": "sha256",
    "encoding": "utf-8",
    "field_order": ["day_book_sha256", "arm", "domain_separator"],
    "joiner": "|",
    "domain_separator": "P003_GATE1_MULTIDAY",
    "hex_truncation_chars": 8,
    "int_base": 16,
    "expression": ("int(sha256(f'{day_book_sha256}|{arm}|"
                   "P003_GATE1_MULTIDAY'.encode()).hexdigest()[:8], 16)"),
    "why_the_truncation_is_declared": (
        "8 hex characters is a 32-bit seed. It is not a safety property and "
        "it is not arbitrary -- it is the number that fixes WHICH integer "
        "the RNG is started from, and a silent change to 16 would reseed "
        "every arm-day while every other field in the receipt read the same"),
    "what_it_pins": "THE DATA, not merely the RNG (R-234 / protocol rule 10): "
                    "the book digest is an input, so a book that moved cannot "
                    "reuse a draw sequence",
    "implemented_by": "de_multiday_gate1_runner.seed_for()",
    "checked_by": "the battery drives seed_from_convention() against "
                  "seed_for() and against one mutation per declared field",
}

#: R-572(B)(1) -- WHERE THE DRAWS COME FROM. Unruled until R-572: nothing in
#: the repository called BE's cascade to produce the null, so the provenance
#: block could have been GENERATED or SUPPLIED and the artifact would look the
#: same either way.
DRAW_SOURCE_RULE = {
    "ruling": "R-572(B)(1) (coordinator)",
    "on_a_real_day": "GENERATED_IN_PROCESS",
    "how": "the runner imports be_cancel_axis_null, verifies the digest of "
           "the file the import actually loaded (module.__file__), and calls "
           "its draw_null() itself",
    "why_in_process": "generation in the SAME process is what binds the "
                      "verified digest to the numbers. A digest verified "
                      "here and draws handed over from elsewhere says which "
                      "cascade EXISTS, not which one produced these draws "
                      "(reviewer efba2b6 item 2)",
    "supplied_draws": "FIXTURE ONLY. A supplied draw set on a ruled day "
                      "REFUSES",
    "the_door_is_shut_structurally_not_on_the_callers_word": (
        "fixture mode is refused when the day is IN the ruled set, and "
        "real mode is refused when the day is NOT -- the same lock DE 76 "
        "put on `fixture=True` in de_data_root, in the second place a "
        "caller could have walked through on its own word"),
    "de_does_not_reimplement_the_cascade": True,
}

#: R-572(B)(2) -- the not-before date governs the READ, not the runs.
TIMING_RULE = {
    "ruling": "R-572(B)(2) (coordinator)",
    "read_not_before_utc": "2026-09-09T00:06:00Z",
    "what_that_date_governs": "the AGGREGATE READ only -- the unsealing of "
                              "the economic fields and the section-7 verdict",
    "day_runs_allowed_for_closed_qualifying_days": True,
    "why": "a per-day run is SEALED: it publishes resources, counts and "
           "statuses and no economic field, so running 09-03 today cannot "
           "inform any later choice. Holding the runs behind the aggregate "
           "date bought nothing and cost the calendar",
    "what_still_refuses": ["a day that is not closed", "a day that does not "
                           "qualify on the four conjuncts and day-quality",
                           "a day outside the ruled set",
                           "the aggregate read before the date",
                           "the aggregate read before all G days are "
                           "complete"],
    "superseded_field": "run_not_before_utc (params v1), a whole-set property "
                        "that read as a bar on every run",
}

#: R-572(B)(3) -- the sealed/unsealed layout, made SYMMETRIC.
SEAL_LAYOUT = {
    "ruling": "R-572(B)(3) (coordinator)",
    "the_asymmetry_it_fixes": (
        "a sealed artifact carried `sealed_at_every_depth` and "
        "`sealed_field_names`; an unsealed one dropped BOTH and gained "
        "`economic`. A consumer keying on either key got None after "
        "unsealing and could not tell 'unsealed' from 'a field I misspelled'"),
    "keys_present_in_BOTH_states": ["sealed", "seal_status",
                                    "sealed_at_every_depth",
                                    "sealed_field_names", "economic"],
    "sealed_values": {"sealed": True, "sealed_at_every_depth": True,
                      "sealed_field_names": "the full ECONOMIC_FIELDS list",
                      "economic": "ABSENT -- the one key that is absent by "
                                  "design, because a field a reader can see "
                                  "is a field a reader can quote"},
    "unsealed_values": {"sealed": False, "sealed_at_every_depth": False,
                        "sealed_field_names": "[] -- nothing is sealed",
                        "economic": "present; None with a stated reason on a "
                                    "refused arm-day"},
    "why_economic_is_the_exception": (
        "symmetry of KEYS is for the consumer; the seal is for the reader. "
        "Present-and-null would leak the shape of what is sealed and invites "
        "a reader to quote a null as a result. So four keys are symmetric by "
        "value and the fifth is symmetric by RULE: present iff unsealed, "
        "stated in both states"),
    "consumer_falsifier": "de_multiday_gate1_runner."
                          "assert_seal_layout_symmetric() -- driven on BOTH "
                          "states, with the pre-fix layout as the known-bad",
}


#: R-572(B)(4), second convention -- WHAT `--dry-run-ledger` ACTUALLY
#: COVERS. It reads day verdicts and the declared read-state table. That is
#: ALL it reads, and the receipt it writes is green when the ledger is
#: healthy no matter what state anything else is in. The playbook's P2, P6
#: and P7 -- BE's book, the pinned models and thetas, BE's cascade digest --
#: are NOT in its scope, so a green dry run says nothing about any of them.
#: Declared here because the gap is invisible at the console: the run exits
#: 0 and prints a receipt.
#:
#: The runner keeps its OWN literals in `dry_run_ledger()`; the battery
#: compares the two lists. Sharing one constant would make the check unable
#: to fail (rule 16).
DRY_RUN_LEDGER_SCOPE_DECLARED = {
    "entry_point": "de_multiday_gate1_runner.py --dry-run-ledger",
    "reads": ["day-verdict files", "the declared read-state table"],
    "does_NOT_read": ["any reference book", "any arm", "any score stream",
                      "any economics"],
    "does_NOT_verify": {
        "P2_BE_reference_book": "verify_day_inputs() is not called",
        "P6_pinned_models_and_thetas": "verify_pinned_models() and "
                                       "verify_pinned_thetas() are not "
                                       "called",
        "P7_BE_cascade_module_digest": "verify_be_module() is not called",
    },
    "therefore": "a GREEN dry run is evidence about the LEDGER and the ruled "
                 "day set, and about nothing else. It is a before-picture, "
                 "not a preflight",
    "what_it_writes": "a dry-run ledger receipt; no book, no arm, no "
                      "economics, no seal",
    "checked_by": "the battery compares this declared scope against the "
                  "runner's own what_this_reads / what_this_does_NOT_read",
}


def seed_from_convention(day_book_sha256: str, arm: str,
                         conv: dict | None = None) -> int:
    """REFERENCE implementation, built from the DECLARED fields only.

    Deliberately not imported by the runner: the runner keeps its own
    expression, and the battery compares the two. One shared constant would
    make the check unable to fail, which is rule 16's exact defect."""
    import hashlib as _h
    c = SEED_CONVENTION if conv is None else conv
    parts = {"day_book_sha256": day_book_sha256, "arm": arm,
             "domain_separator": c["domain_separator"]}
    payload = c["joiner"].join(parts[f] for f in c["field_order"])
    if c["hash"] != "sha256":
        raise DesignRefused(
            f"REFUSED: the declared hash is {c['hash']!r}; this reference "
            f"implements sha256 only, and silently substituting another is "
            f"how a declared convention drifts from its code")
    digest = _h.sha256(payload.encode(c["encoding"])).hexdigest()
    return int(digest[:c["hex_truncation_chars"]], c["int_base"])


def verify_declaration_chain(root: Path | None = None,
                             chain=None) -> dict:
    """Every chain entry's digest READ from the file, never trusted typed.

    A chain of typed digests is a chain of claims. This round typed one that
    was invented outright and one that was a literal `PLACEHOLDER`; both were
    caught by computing them instead."""
    base = (Path(root) if root is not None
            else Path(DR.resolve()["data_root"])) / "pm_5min/derived"
    rows, bad = [], []
    for name, sha in (DECLARATION_CHAIN if chain is None else chain):
        p = base / name
        if not p.is_file():
            bad.append({"file": name, "why": "ABSENT"})
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append({"file": name, "declared": sha, "computed": got,
                     "agrees": got == sha})
        if got != sha:
            bad.append({"file": name, "declared": sha, "computed": got})
    if bad:
        raise DesignRefused(
            f"REFUSED: the supersession chain does not match the artifacts "
            f"on disk: {bad}. A typed digest in a chain is a claim, and this "
            f"chain already carried an invented one.")
    return {"n_entries": len(rows), "every_digest_recomputed": True,
            "entries": rows}


#: R-555 (USER), the ruled day set. Named here so the declaration STATES
#: it rather than leaving a reader to derive it -- and the battery asserts
#: it equals the set the runner loads from the parameter file.
RULED_DAYS = ("2026-09-03", "2026-09-04", "2026-09-05",
              "2026-09-06", "2026-09-07", "2026-09-08")
RULED_G = len(RULED_DAYS)


class DesignRefused(RuntimeError):
    """The declared design cannot be honoured on the inputs given."""


# ---------------------------------------------------------------- the rule

def per_day_location(observed: float, null_draws: list) -> dict:
    """One-sided, LARGER IS BETTER: p = (1 + #{null >= observed}) / (1+K)."""
    k = len(null_draws)
    if k < MIN_DRAWS:
        raise DesignRefused(
            f"REFUSED: {k} draws is below the declared minimum {MIN_DRAWS} "
            f"(rule 6). An under-sampled correct null flatters as much as "
            f"a wrong one.")
    ge = sum(1 for v in null_draws if v >= observed)
    return {"n_draws": k, "n_null_ge_observed": ge,
            "p_one_sided": (1 + ge) / (1 + k), "floor": 1 / (1 + k)}


def per_day_standardised_excess(observed: float, null_draws: list) -> float:
    """Z = (observed - mean(null)) / sd(null). The per-day cluster value."""
    sd = statistics.pstdev(null_draws)
    if sd == 0:
        raise DesignRefused(
            "REFUSED: the null has zero dispersion, so a standardised "
            "excess is undefined. A degenerate null is a STATUS, never a "
            "large Z.")
    return (observed - statistics.fmean(null_draws)) / sd


def _smallest_G(alpha: float, m: int, cap: int = 40) -> int | None:
    """The smallest number of day clusters whose unanimous sign test clears
    Holm -- COMPUTED, because a hardcoded 7 was wrong by one and the whole
    point of the field is to price a decision in calendar days."""
    thr = alpha / m
    for g in range(1, cap + 1):
        if 2.0 ** (-g) <= thr:
            return g
    return None


def day_cluster_verdict(z_by_day: dict, *, alpha: float = ALPHA,
                        m: int = MULTIPLICITY, g: int | None = None) -> dict:
    """THE SECTION-7 PREDICATE, DECLARED BEFORE ANY DAY IS SEEN.

    Cluster unit is the UTC day (rule 8). The statistic is the mean of the
    per-day standardised excesses; the test is the EXACT SIGN TEST over the
    G days, whose one-sided p is 2^-G when every day agrees.

    THE ASYMMETRY IS DELIBERATE AND IS THE POINT. This is a STOPPING rule:
      FAIL  -- cheap, and needs no significance. An arm fails if its
               cluster mean is <= 0, or if its day signs are not unanimous.
               "Did not beat the null" is not a claim that needs power.
      PASS  -- expensive, and CAPPED BY ARITHMETIC AT THIS G. With G = 5 the
               smallest attainable one-sided sign-test p is 2^-5 = 0.03125,
               and Holm at m = 2 compares the smaller p against alpha/2 =
               0.025. 0.03125 > 0.025, SO NO ARM CAN CLEAR HOLM ON THIS RUN
               EVEN IF EVERY DAY GOES ITS WAY. A pass is therefore
               DIRECTIONAL AND CONSISTENT, NEVER SIGNIFICANCE-BEARING --
               the same limit R-529(A) ruled for the forward race, declared
               here BEFORE the run rather than discovered after it.
    COMPUTED, not asserted: the smallest G that clears Holm is 6 at m = 2
    (2^-6 = 0.015625 <= 0.025) and 5 at m = 1 (2^-5 = 0.03125 <= 0.05).
    So ONE MORE ADMISSIBLE DAY would make a unanimous pass
    significance-bearing at m = 2 -- which is a fact worth having before
    the run rather than after it."""
    # G IS BOUND AT RUN TIME FROM THE RULED DAY SET, not from a constant.
    # R7 computes two candidate sets and the USER's answer selects one; the
    # module default is v1's five only so the rule can be driven on a
    # fixture before that answer exists.
    g = G if g is None else g
    days = sorted(z_by_day)
    if len(days) != g:
        raise DesignRefused(
            f"REFUSED: the cluster test is declared over exactly G = {g} "
            f"days; got {len(days)}. Dropping or adding a day after the "
            f"fact is choosing after seeing (rule 11).")
    zs = [z_by_day[d] for d in days]
    mean_z = statistics.fmean(zs)
    n_pos = sum(1 for z in zs if z > 0)
    unanimous = n_pos == g
    p_sign = 2.0 ** (-g) if unanimous else None
    holm_threshold = alpha / m
    clears_holm = bool(p_sign is not None and p_sign <= holm_threshold)
    fails = (mean_z <= 0) or (not unanimous)
    return {
        "days": days, "z_by_day": {d: z_by_day[d] for d in days},
        "cluster_unit": "UTC day", "G": g,
        "mean_standardised_excess": mean_z,
        "n_days_positive": n_pos, "signs_unanimous": unanimous,
        "p_one_sided_sign_test": p_sign,
        "multiplicity_m": m, "alpha": alpha,
        "holm_threshold_for_the_smaller_p": holm_threshold,
        "clears_holm": clears_holm,
        "best_attainable_p_at_this_G": 2.0 ** (-g),
        "a_pass_is_significance_bearing": clears_holm,
        "FAILS_THE_SECTION_7_PREDICATE": fails,
        "verdict": ("FAILS_TO_BEAT_THE_REPLAY_NULL" if fails
                    else "BEATS_DIRECTIONALLY_NOT_SIGNIFICANTLY"),
        "smallest_G_that_clears_holm": _smallest_G(alpha, m),
        "smallest_G_that_clears_holm_at_m_1": _smallest_G(alpha, 1),
        "why_a_pass_cannot_be_significant_here": (
            f"2^-{g} = {2.0 ** (-g)} against a Holm threshold of "
            f"{holm_threshold} at m = {m}; the smallest G that clears is "
            f"{_smallest_G(alpha, m)} at m = {m} and "
            f"{_smallest_G(alpha, 1)} at m = 1"),
    }


def arm_day_admissible(n_decisions: int, null_draws: list) -> dict:
    """(4) THE DEGENERACY BARS, applied. Declared before any day is seen."""
    sd = statistics.pstdev(null_draws) if null_draws else 0.0
    mean = statistics.fmean(null_draws) if null_draws else 0.0
    reasons = []
    if n_decisions < MIN_DECISIONS_PER_ARM_DAY:
        reasons.append(
            f"decisions {n_decisions} < declared minimum "
            f"{MIN_DECISIONS_PER_ARM_DAY}")
    if sd < SD_FLOOR_FRACTION * abs(mean):
        reasons.append(
            f"null sd {sd:.6g} < {SD_FLOOR_FRACTION} * |mean {mean:.6g}| "
            f"= {SD_FLOOR_FRACTION * abs(mean):.6g}; Z explodes as sd -> 0")
    return {"n_decisions": n_decisions, "null_sd": sd, "null_mean": mean,
            "sd_over_abs_mean": (sd / abs(mean)) if mean else None,
            "admissible": not reasons,
            "status": "OK" if not reasons else "DEGENERATE_ARM_DAY_REFUSED",
            "reasons": reasons}


def verify_book_digest(day: str, book_path: str, declared_sha256: str,
                       actual_sha256: str) -> dict:
    """A day whose reference book does not match its pinned digest REFUSES
    -- the whole day, not the offending draw."""
    if actual_sha256 != declared_sha256:
        raise DesignRefused(
            f"REFUSED: day {day} reference book digest mismatch at "
            f"{book_path}: declared {declared_sha256}, actual "
            f"{actual_sha256}. A day whose book moved is not the day the "
            f"design declared, and no draw on it is admissible.")
    return {"day": day, "book": book_path, "sha256": actual_sha256,
            "verified": True}


# ------------------------------------------------------------ declaration

def day_sets_from_the_ledger(root: Path | None = None) -> dict:
    """(7) THE DAY SET, DERIVED ON QUALITY -- the USER's bar (R-497(F)(1):
    "collector version is NOT a bar; QUALITY is the bar").

    The rule: the four conjuncts AND day-quality AND -- under one reading
    only -- not previously opened for any read. THE PARAMETER THE USER IS
    BEING ASKED TO FILL is whether a day opened for a read of the FROZEN
    CANDIDATE counts as untouched for a Gate-1 test of the ARMS, which are
    a different object. Both resulting sets are computed here with their
    Holm arithmetic; G is fixed before any run."""
    import glob
    import re
    base = Path(root) if root is not None else Path(
        __file__).resolve().parents[2]
    # THE WORKTREE DATA-SHELL TRAP, SECOND INSTANCE. In a seat worktree
    # `<root>/data` is the git-materialised shell holding only COMMITTED
    # artifacts, and the real tree hangs off the R-397 symlink at
    # `<root>/data/data`. Reading the shell silently produced a 3-day
    # qualifying set where the real ledger has 6 -- caught only because the
    # count disagreed with a hand check. Resolve the symlink when it exists
    # and RECORD which root was read.
    # R-559(C): ONE resolution for the whole DE surface, imported.
    if root is None:
        rr = DR.resolve()
        data_root = Path(rr["data_root"])
    else:
        rr = {"branch": "0_explicit_root_argument", "data_root": None}
        data_root = base / "data" / "data" \
            if (base / "data" / "data").exists() else base / "data"
    # (4) IT MUST REFUSE, NOT RETURN AN EMPTY SET. The reviewer drove a
    # non-ledger root and got a silent [] -- a derivation that answers
    # "no days qualify" when it is looking at the wrong tree is worse than
    # one that crashes.
    resolved = data_root.resolve()
    if str(resolved) != DR.CANONICAL_DATA_ROOT:
        raise DesignRefused(
            f"REFUSED: the ledger root resolves to {resolved}, not the "
            f"declared {DR.CANONICAL_DATA_ROOT} (branch {rr['branch']}). "
            f"A day set derived from the "
            f"wrong tree is not a smaller day set, it is a different "
            f"question -- and the failure mode is an EMPTY answer that "
            f"looks like a result.")
    rows = {}
    for path in sorted(glob.glob(str(
            data_root / "pm_5min/derived/da_dayverdict_2026*.json"))):
        if "superseded" in path:
            continue
        found: dict = {}

        def walk(o, dep=0):
            if dep > 6:
                return
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in LEDGER_CONJUNCTS and not isinstance(
                            v, (dict, list)) and k not in found:
                        found[k] = v
                    if isinstance(v, (dict, list)):
                        walk(v, dep + 1)
            elif isinstance(o, list):
                for v in o[:8]:
                    walk(v, dep + 1)
        walk(json.loads(Path(path).read_text()))
        raw = re.search(r"(\d{8})", path).group(1)
        day = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
        rows[day] = {k: found.get(k) for k in LEDGER_CONJUNCTS}
        rows[day]["all_conjuncts_and_quality"] = all(
            found.get(k) is True for k in LEDGER_CONJUNCTS)
    if not rows:
        raise DesignRefused(
            f"REFUSED: no day-verdict files under {resolved}. An empty "
            f"verdict set is a MISSING LEDGER, not a day set of size zero.")
    qualifying = [d for d, v in sorted(rows.items())
                  if v["all_conjuncts_and_quality"]]
    set_a = list(qualifying)
    # SET B IS DERIVED FROM THE FIELD, not from a membership test against a
    # prose dict: a day is excluded iff its `previously_opened_for` is not
    # `none`. A day absent from the table is UNKNOWN, not clean, and
    # refuses rather than defaulting into set B.
    unknown = [d for d in qualifying if d not in DAY_READ_STATE]
    if unknown:
        raise DesignRefused(
            f"REFUSED: {unknown} qualify on the ledger but carry no "
            f"read-state field. A day whose read state is unrecorded "
            f"cannot be declared untouched by omission.")
    set_b = [d for d in qualifying
             if DAY_READ_STATE[d]["previously_opened_for"] == OPENED_NONE]

    def holm(g):
        p = 2.0 ** (-g)
        return {"G": g, "best_attainable_one_sided_p": p,
                "holm_threshold_m2": ALPHA / 2,
                "clears_holm_at_m2": p <= ALPHA / 2,
                "clears_holm_at_m1": p <= ALPHA,
                "smallest_G_that_clears_m2": _smallest_G(ALPHA, 2),
                "smallest_G_that_clears_m1": _smallest_G(ALPHA, 1)}
    return {
        "ledger_root_read": str(data_root),
        "ledger_root_resolved": str(resolved),
        "declared_ledger_root": DR.CANONICAL_DATA_ROOT,
        "root_verified_at_run_time": True,
        "root_resolution": rr,
        "why_the_root_is_recorded": (
            "in a seat worktree `<root>/data` is the git-materialised "
            "shell of COMMITTED artifacts only; the real tree hangs off "
            "the R-397 symlink at `<root>/data/data`. Reading the shell "
            "silently produced a 3-day qualifying set where the ledger has "
            "6. The root that was read travels with the answer"),
        "n_verdict_files_read": len(rows),
        "rule": "the four ledger conjuncts AND day-quality AND (under set "
                "B only) not previously opened for any read",
        "conjuncts": list(LEDGER_CONJUNCTS),
        "version_is_NOT_a_bar": (
            "R-497(F)(1), the USER verbatim: 'collector version is NOT a "
            "bar; QUALITY is the bar'. R-547(C)'s 'only era-pure clob_v4_1 "
            "days' imported a bar the USER never set, and v1 of this "
            "declaration inherited it. WITHDRAWN HERE."),
        "ledger_rows_as_read": rows,
        "qualifying_on_quality": qualifying,
        "day_read_state": {d: DAY_READ_STATE[d] for d in qualifying},
        "set_b_rule": "a day is in SET B iff its `previously_opened_for` "
                      "field reads `none`; a day with no field REFUSES "
                      "rather than defaulting into it",
        "withdrawn_sentence": {
            "text": "the race scored a different object on these days, "
                    "sealed and unread",
            "carried_in": ["design v1", "design v2"],
            "withdrawn_by": "R-549(A); RESULTS.md:681",
            "why": "09-01 and 09-02 were OPENED under the interim read and "
                   "are CONSUMED, so 'unread' was false for two of the "
                   "five days v1 named",
            "what_survives_of_it": "the thetas were fixed on the consumed "
                                   "08-24 hour and NOTHING ABOUT THE ARMS "
                                   "was chosen on any of these days -- "
                                   "which is the half that bears on rule "
                                   "11, and it is now stated on its own"},
        "THE_PARAMETER_FOR_THE_USER": (
            "do days previously opened for a read of the FROZEN CANDIDATE "
            "count as UNTOUCHED for a Gate-1 test of the ARMS? The arms "
            "are a different object with thetas fixed on the consumed "
            "08-24 hour, and no arm score has been read on any of these "
            "days -- but the tape has been looked at."),
        "SET_A_reads_count_as_untouched": {
            "days": set_a, "holm": holm(len(set_a))},
        "SET_B_reads_consume_the_day": {
            "days": set_b, "holm": holm(len(set_b)),
            "derived_from_the_field": True},
        "what_the_answer_decides": (
            "SET A gives G = 6, which CLEARS Holm at m = 2 -- a "
            "significance-bearing answer is possible. SET B gives G = 3, "
            "which clears at NEITHER m = 2 NOR m = 1. The parameter is the "
            "difference between a run that can settle section 7 with a "
            "p-value and one that can only ever be directional."),
        "accrual_schedule": {
            "2026-09-06": "OPEN at declaration time (day_closed_calendar "
                          "false); closes 00:00Z 09-07 and is verdicted at "
                          "00:06Z 09-07",
            "rule": "each further day is verdicted at 00:06Z the following "
                    "day, so set A reaches G = 7 on 09-07 and set B "
                    "reaches G = 4",
            "G_is_fixed_before_any_run": True},
    }


#: (addendum 2) measured seconds, and the arithmetic done here rather than
#: in prose. v1 said "~2 hours of null per ARM-day"; 290.9 s is one hour
#: for BOTH arms, so the extrapolation is per DAY for both arms and v1
#: overstated the null by a factor of two.
MEASURED = {"de_arms_replay_one_hour_s": 47.0,
            "de_arms_replay_peak_gb": 0.61,
            "be_null_500_draws_one_hour_BOTH_arms_s": 290.9,
            "windows_in_the_measured_hour": 12,
            "windows_in_a_utc_day": 288}
HEAVY_RUN_WRAPPER = (
    "flock -n /home/yuqing/ctaNew/data/.heavy_run.lock "
    "systemd-run --user --scope --slice=research.slice "
    "-p MemoryMax=8G -p CPUQuota=100% <cmd>")


def _resources() -> dict:
    x = MEASURED["windows_in_a_utc_day"] / MEASURED[
        "windows_in_the_measured_hour"]
    null_day_s = MEASURED["be_null_500_draws_one_hour_BOTH_arms_s"] * x
    replay_day_s = MEASURED["de_arms_replay_one_hour_s"] * x
    def total(g):
        return {"G": g,
                "null_hours": g * null_day_s / 3600.0,
                "replay_hours": g * replay_day_s / 3600.0,
                "sequential_cpu_hours": g * (null_day_s + replay_day_s)
                / 3600.0}
    return {
        "basis": "measured on the consumed 08-24 hour",
        "measured": MEASURED,
        "extrapolation_factor": x,
        "per_day_BOTH_arms": {"null_s": null_day_s,
                              "null_hours": null_day_s / 3600.0,
                              "replay_s": replay_day_s,
                              "replay_hours": replay_day_s / 3600.0},
        "totals": {"G5": total(5), "G6": total(6)},
        "v1_mislabelled_this": (
            "v1 read 290.9 s as one hour for ONE arm and wrote '~2 hours "
            "of null per ARM-day'. It is one hour for BOTH arms, so the "
            "figure is ~1.94 h per DAY for both arms and v1 overstated the "
            "null by a factor of two. Corrected here as computed fields "
            "from the measured seconds"),
        "the_estimate_is_an_ESTIMATE": (
            "a linear 24x extrapolation from one hour; the day-1 smoke "
            "replaces it with a measurement"),
        "cap": "one CPU, MemoryMax=8G, never raised (R-174); if a day "
               "exceeds the cap the day REFUSES rather than the cap rising",
        "mandatory_wrapper": HEAVY_RUN_WRAPPER,
        "wrapper_rule": (
            "SEAT_PROTOCOL rule 20: every heavy step runs under this "
            "wrapper. The flock REFUSES if another heavy run holds the "
            "lock, so two seats cannot contend for the same 8G"),
        "mandatory_first_run": (
            "the ONE-DAY SMOKE -- economic fields SEALED, resource "
            "observation published -- before any further day is run"),
    }


#: Paths at which the withdrawn phrase may appear -- because it is being
#: WITHDRAWN there, not asserted. A grep at the artifact finds the string;
#: this field says where and why, and REFUSES an occurrence anywhere else.
WITHDRAWN_PHRASE = "sealed and unread"
WITHDRAWAL_CONTEXTS = (
    "R7_the_day_set.withdrawn_sentence.text",
    "days.nothing_about_the_ARMS_has_been_chosen_on_them",
    "supersedes.v3_closes[0]",
)


def _withdrawn_phrase_audit(payload: dict) -> dict:
    """Every occurrence of the withdrawn phrase must be a WITHDRAWAL."""
    found = []

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, str) and WITHDRAWN_PHRASE in o:
            found.append(path)
    walk(payload)
    stray = [f for f in found if f not in WITHDRAWAL_CONTEXTS]
    if stray:
        raise DesignRefused(
            f"REFUSED: the withdrawn phrase {WITHDRAWN_PHRASE!r} appears "
            f"outside a declared withdrawal context at {stray}. A grep at "
            f"the artifact must find it ONLY where it is being withdrawn.")
    return {"phrase": WITHDRAWN_PHRASE, "n_occurrences": len(found),
            "occurrences": found,
            "all_in_a_withdrawal_context": True,
            "why_it_appears_at_all": "a sentence cannot be withdrawn "
                                     "without being named; the audit is "
                                     "what separates naming it from "
                                     "asserting it",
            "declared_contexts": list(WITHDRAWAL_CONTEXTS)}


def _r8_from_resources() -> dict:
    """(3) R8's estimate is a REFERENCE to `resources.totals`, computed --
    never a typed sentence that can drift from the field beside it."""
    r = _resources()
    return {
        "estimate_source": "resources.totals -- COMPUTED from the measured "
                           "seconds, not typed",
        "sequential_cpu_hours": {
            "G5": r["totals"]["G5"]["sequential_cpu_hours"],
            "G6": r["totals"]["G6"]["sequential_cpu_hours"]},
        "null_hours": {"G5": r["totals"]["G5"]["null_hours"],
                       "G6": r["totals"]["G6"]["null_hours"]},
        "the_estimate_is_an_ESTIMATE": r["the_estimate_is_an_ESTIMATE"],
        "the_500_draw_minimum_is_protected_by": "REFUSING THE ARM-DAY",
        "never_by": ["lowering the draw count", "raising the cap",
                     "sampling fewer days"],
        "why_written_down": "the cap is protected and the DRAW COUNT was "
                            "not; the tempting response to a time overrun "
                            "is to cut draws, and that response is "
                            "forbidden in advance",
        "agrees_with_resources_totals": True,
    }


def carrying_commit_block(producing: Path) -> dict:
    """R-387's `carrying_commit`, with the property that actually matters.

    A whole-tree `dirty` flag is too coarse and too easy to satisfy: what a
    reader needs is whether THE FILE THAT RAN is the file the named commit
    holds. So the producer's own blob at HEAD is compared to its bytes on
    disk. `tree_dirty` is reported beside it and is NOT the check -- other
    seats' files being uncommitted says nothing about this producer.

    DA's rule, in DE's emitters: "a carrying_commit recorded over a dirty
    tree points at bytes that did not run"."""
    import subprocess
    root = Path(__file__).resolve().parents[2]

    def _git(*a, raw=False):
        r = subprocess.run(["git", "-C", str(root), *a],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return None
        # `raw` matters: `.strip()` eats the trailing newline of a blob and
        # then EVERY file compares unequal to itself. Caught by the positive
        # control, which is what a positive control is for.
        return r.stdout if raw else r.stdout.strip()

    head = _git("rev-parse", "HEAD")
    rel = str(producing.resolve().relative_to(root))
    blob = _git("show", f"HEAD:{rel}", raw=True)
    on_disk = producing.read_text()
    status = _git("status", "--porcelain")
    return {
        "carrying_commit": head,
        "producing_code_path": rel,
        "producing_code_is_the_committed_bytes": (blob is not None
                                                  and blob == on_disk),
        "tree_dirty": bool(status) if status is not None else None,
        "tree_dirty_is_NOT_the_check": (
            "other seats' uncommitted files say nothing about this "
            "producer; the check above compares THIS file's blob at HEAD "
            "with the bytes that ran"),
    }


def declaration() -> dict:
    return {
        "protocol": PROTOCOL,
        "status": "DESIGN_DECLARATION_NO_DATA_TOUCHED",
        "declared_before_any_draw": True,
        "ruling": "R-547 (USER). Gate 1's control is the REPLAY NULL; the "
                  "five named days decide the section-7 stopping rule; "
                  "design and null committed before data.",
        "days": {
            "STATUS": "RULED BY R-555 (USER). v1's five named days rested "
                      "on the imported era bar and are superseded; R7's "
                      "candidate sets are superseded too -- the ruled set "
                      "is NEITHER of them",
            # v8: THIS FIELD READ `True` UNTIL THIS ROUND, four hours after
            # the USER answered. A declaration that still says its own key
            # parameter is pending is a declaration a cold reader cannot
            # act on, and the CLI summary was printing the stale value.
            "G_is_PENDING_the_USER_parameter": False,
            "G_RULED": RULED_G,
            "ruled_set": list(RULED_DAYS),
            "ruling": "R-555 (USER): untouched days only. Days opened for a "
                      "read of the frozen candidate (08-29, 09-01, 09-02) "
                      "do NOT count as untouched",
            "the_ruled_set_is_neither_candidate": (
                "SET A held 08-29/09-01/09-02, which the ruling excludes; "
                "SET B held the three untouched days that EXISTED on "
                "09-06 and G would have been 3. The ruling keeps SET B's "
                "predicate and extends it forward to 09-06, 09-07 and "
                "09-08 as each closes and is verdicted -- so G is 6 and "
                "the set completes at the 2026-09-09T00:06Z verdict"),
            "if_a_future_day_fails_its_verdict": (
                "the set is NOT extended by choosing another day: it waits "
                "for the next qualifying CLOSED day in calendar order and "
                "G stays 6 (R-555(B))"),
            "v1_named": list(DAYS), "v1_G": G,
            "why_these": "the only era-pure clob_v4_1 days in existence "
                         "(R-547(C)); 08-29/30/31 straddle era boundaries "
                         "and are inadmissible; 08-20..08-25 are consumed",
            "admissibility_source": "da_dayverdict_<YYYYMMDD>.json, the "
                                    "era-admission block -- NOT CLAUDE.md "
                                    "rule 5's mm_hf Binance boundary, "
                                    "which does not govern this tape "
                                    "(R-547(B))",
            "nothing_about_the_ARMS_has_been_chosen_on_them": (
                "the thetas were fixed on the consumed 08-24 hour and no "
                "arm score has been read on any of these days. THE "
                "STRONGER CLAIM v1 AND v2 MADE -- that the tape itself was "
                "sealed and unread -- IS WITHDRAWN (R-549(A)); see "
                "R7_the_day_set.withdrawn_sentence and the per-day "
                "`previously_opened_for` field"),
        },
        "arms": list(ARMS),
        "theta": THETA,
        "theta_pins": THETA_PINS,
        "theta_is_not_refitted_on_any_of_the_five_days": True,
        "supersedes": {
            # THE LAST ELEMENT OF THE CHAIN, never a typed constant. v7 named
            # v2 here and a reader resolving this field walked past v3..v6.
            "path": f"data/pm_5min/derived/{DECLARATION_CHAIN[-1][0]}",
            "sha256": DECLARATION_CHAIN[-1][1],
            "chain": [list(x) for x in DECLARATION_CHAIN],
            "version": VERSION,
            "the_version_travels_in_ONE_place": (
                "VERSION; the protocol suffix, the emitted filename and the "
                "head of this chain are derived from it and the battery "
                "asserts all three agree. v7 on disk read protocol V4, "
                "filename v7 and supersedes v2 -- three axes, three answers"),
            "v1_and_v2_untouched": True,
            "v3_closes": [
                "the withdrawn 'sealed and unread' sentence, replaced by a "
                "per-day `previously_opened_for` field that R7 derives "
                "SET B from (reviewer 41cba2c A.2)",
                "R8's typed '~20 hours' estimate, replaced by a computed "
                "reference to resources.totals",
                "the worktree data-shell trap, recorded as a field with "
                "the root this emission read"],
            "why": "the reviewer's be03d4d found two blocking items and "
                   "seven places a choice could still be made after seeing "
                   "a day. Every one is closed here as a FIELD",
            "defects_in_v1_this_names": [
                "the imported era bar -- v1 said 'the only era-pure "
                "clob_v4_1 days', which R-497(F)(1) rules out: version is "
                "not a bar, QUALITY is. WITHDRAWN",
                "the battery count -- v1's receipt recorded n_checks 19 "
                "against the source's EXPECTED_CHECKS 20, because main() "
                "wrote EXPECTED_CHECKS - 1 rather than the count the run "
                "produced. The source hash matched: a RECEIPT-COUNT "
                "defect, not a code difference",
                "the resource arithmetic -- v1 read 290.9 s as one arm-hour "
                "and wrote '~2 hours of null per ARM-day'; it is one hour "
                "for BOTH arms, so v1 overstated the null by a factor of "
                "two",
                "the smoke -- v1 published day 1's economic result before "
                "days 2..G were run, leaving early stopping reachable"]},
        "R1_asm_the_scored_book": {
            "blocking_in_v1": True,
            "verified_by_DE_at_the_code": (
                "be_cancel_axis_null.py:188-192 -- `ref, asm = "
                "c['fr']['reference'], c['asm']` then `scored = "
                "asm['by_arm'][(COIN, ARMS['CONDVALUE_X_SKEW']['head'])][0]` "
                "and rows are the generations IN `scored`. A book without "
                "`asm` raises on c['asm'] and there is no decision "
                "population at all"),
            "what_asm_must_contain_per_day": {
                "by_arm": "a mapping keyed by (coin, head) for BOTH pinned "
                          "heads -- q1_arrival_composed_lgbm and "
                          "incumbent_linear_d -- whose [0] element is the "
                          "set of scored (slug, side, t0) generation keys",
                "scored_at": "the arms' PINNED thetas, not refitted ones",
                "coverage": "every generation the reference carries for "
                            "that day, so a generation absent from `scored` "
                            "is a scorer-coverage fact and not a silent "
                            "drop"},
            "how_its_digest_enters_the_seed": (
                "the seed is derived from the digest of the WHOLE day book "
                "as BE publishes it, which contains `asm`; so a book whose "
                "score stream changed cannot reuse a draw sequence. The "
                "declaration additionally requires BE to publish "
                "sha256(asm) separately, so a reader can tell a "
                "score-stream change from a reference change"),
            "discharge_on_one_day_first": (
                "the 'same cascade' premise is currently a live refutation "
                "condition, not a hypothetical. BE emits day 1's book with "
                "`asm` and the null is driven on it BEFORE the other four "
                "are built"),
        },
        "R2_the_draw_pool": {
            "as_the_loader_stands": "ONE SHARED scored pool, built from "
                                    "CONDVALUE's head only, used for both "
                                    "arms (be_cancel_axis_null.py:189)",
            "declared_choice": "SHARED",
            "why": (
                "ruling v2 (R-548(C)) separates the NULL from the "
                "DENOMINATOR. The null is per arm because the DRAW is "
                "matched to that arm's own decision count and side split; "
                "the POOL those draws come from is the set of generations "
                "a decision could have been made on, which is a property "
                "of the book and not of the arm. Making the pool per-arm "
                "would make each arm's random cancel land in a different "
                "universe and the two nulls incomparable -- the same "
                "category error as a per-arm shared denominator, in the "
                "other direction"),
            "the_cost_stated": (
                "HAZARD's decisions are drawn from a pool defined by "
                "CONDVALUE's head. If the two heads score materially "
                "different generation sets, HAZARD's null is drawn from a "
                "pool its own head would not have produced. THE DESIGN "
                "REQUIRES BE TO PUBLISH |scored(head)| PER HEAD PER DAY and "
                "their overlap, so the size of that cost is measured "
                "rather than assumed"),
            "refuses_if": f"the two heads' scored sets overlap by less "
                          f"than {HEAD_OVERLAP_FLOOR} of the smaller -- a "
                          f"REFUSAL of the arm-day, not a note",
            "floor_calibration": {
                "measured_on": "the CONSUMED 2026-08-24 13:50-14:50Z hour, "
                               "the one population already seen -- the "
                               "same basis R4's 0.25 was set on",
                "measurement": HEAD_OVERLAP_MEASUREMENT,
                "observed_overlap": HEAD_OVERLAP_OBSERVED,
                "floor": HEAD_OVERLAP_FLOOR,
                "floor_admits_the_consumed_hour":
                    HEAD_OVERLAP_FLOOR <= HEAD_OVERLAP_OBSERVED,
                "what_the_measurement_says": (
                    "the two heads score the SAME 29,813 generations -- "
                    "identical sets, Jaccard 1.0. On the precedent run the "
                    "shared-pool choice costs EXACTLY NOTHING, because "
                    "there is only one pool to share"),
                "why_this_floor_cannot_be_read_as_tuned": (
                    "the observed value is 1.0, which is the CEILING of "
                    "the statistic -- an overlap cannot exceed 1. So no "
                    "floor below 1.0 can have been chosen to clear the "
                    "seen value, and the ONLY floor that would exclude the "
                    "consumed hour is one strictly above 1.0, which is "
                    "unattainable. 0.90 leaves day-to-day headroom without "
                    "being fitted to anything"),
                "and_it_is_a_ONE_DAY_measurement": (
                    "one hour of one coin. Two heads agreeing perfectly "
                    "there does not establish they agree on an unseen day, "
                    "which is exactly why the floor is a per-arm-day "
                    "REFUSAL and not an assumption"),
            },
        },
        "R3_the_coin_set": {
            "declared": ["btc"],
            "why_btc_only": (
                "the pinned thetas ARE btc thetas: CONDVALUE's comes from "
                "lgbm_thresholds_btc.json (0fa2f1f7a5a4c58f) and HAZARD's "
                "from linear_d_btc.json (18701008c2bd18c6). DE verified "
                "that eth siblings EXIST in the same fits directory "
                "(lgbm_thresholds_eth.json ce67009e38a07e8c, linear_d_eth."
                "json fb371f6352214a92) and that they are DIFFERENT "
                "artifacts. Running eth would mean a second (coin, theta) "
                "pair -- a different frozen object, and a different "
                "multiplicity"),
            "what_a_btc_only_run_measures": (
                "whether these two arms, at these two btc thetas, beat a "
                "replay null ON BTC. It says nothing about eth and must "
                "not be reported as a venue-level or programme-level "
                "result"),
            "what_an_eth_arm_would_cost": "m goes from 2 to 4, so the Holm "
                                          "threshold halves to 0.0125 and "
                                          "the smallest clearing G rises "
                                          "from 6 to 7",
        },
        "R4_degeneracy_bars": {
            "min_decisions_per_arm_day": MIN_DECISIONS_PER_ARM_DAY,
            "why_30": (
                "below ~30 matched draws the side split cannot be honoured "
                "without repetition and the null's own dispersion is "
                "dominated by the discreteness of the count. The number is "
                "declared NOW, before any day, precisely so nobody decides "
                "after seeing whether a day with four decisions counts"),
            "sd_floor_fraction": SD_FLOOR_FRACTION,
            "sd_rule": (
                "refuse the arm-day when sd(null) < f * |mean(null)| with "
                "f = 0.25. Z_d = (D - mean)/sd explodes as sd -> 0, and the "
                "v1 degenerate-null falsifier covered only sd == 0 exactly"),
            "calibration_from_the_consumed_hour": (
                "HAZARD's null had sd 0.16234 against mean 0.40033, a "
                "ratio of 0.4055 -- ABOVE the 0.25 floor, so the bar as "
                "declared would have admitted the consumed hour. That is "
                "stated so the floor cannot be read as chosen to exclude "
                "something already seen"),
            "a_refused_arm_day_is_a_STATUS": (
                "it does not shrink G silently; the arm cannot be "
                "aggregated and the run reports the arm as UNTESTABLE on "
                "the declared day set"),
        },
        "R5_the_smoke_is_sealed": {
            "day_1_publishes": ["resource observations", "population counts",
                                "refusal statuses"],
            "day_1_does_NOT_publish": ["D(E0)", "D(E-R)", "Z", "any per-day "
                                       "location", "any null summary "
                                       "statistic"],
            "sealed_until": "all G days are complete",
            "all_G_days_run_regardless_of_interim_results": True,
            "why": (
                "whoever runs days 2..G must not have seen day 1's Z. v1 "
                "said the smoke publishes its resource observation and did "
                "not say the economic fields are withheld, which left "
                "early stopping reachable"),
            "artifact_level_refusal": (
                "the per-day emitter REFUSES to write any economic field "
                "while `n_days_complete < G`; the sealed fields are absent "
                "from the artifact, not present-and-ignored"),
            "IT_IS_NOW_A_CODE_PATH_NOT_A_PROMISE": {
                "emitter": "de_multiday_gate1_runner.seal()",
                "guard": "de_multiday_gate1_runner.assert_no_economic_leak()",
                "shared_name_list": "de_multiday_gate1_runner."
                                    "ECONOMIC_FIELDS -- the emitter and "
                                    "the guard read ONE list and ONE "
                                    "traversal, so they cannot disagree",
                "proof_it_fired": (
                    "the guard caught the emitter on its first run: "
                    "sealing only the top-level `economic` block left "
                    "`admissibility.null_sd` and `null_mean` behind, "
                    "because the R4 block carries null statistics"),
                "fixture_receipt_field": "per_day_sealed_artifacts[*]."
                                         "seal_status and .sealed",
                "fixture_receipt": "p003_de_multiday_gate1_fixture_run_v2"
                                   "__20260906T041609Z.json",
                "what_that_field_shows": "six per-day artifacts, sealed "
                                         "True until the last day and "
                                         "False after -- both states "
                                         "present in one receipt",
            },
        },
        "R6_runtime_verification": {
            "verified_at_run_time_not_merely_recorded": True,
            "what": ["each arm's theta against its pinned JSON path",
                     "each model artifact's sha256 against the pinned "
                     "digest"],
            "on_mismatch": "REFUSE the run -- not the day, the RUN, because "
                           "a moved model means the object under test is "
                           "not the frozen one",
            "why_not_a_record": "a recorded digest that nobody compares is "
                                "provenance theatre; it makes a refit "
                                "detectable and does not detect it",
        },
        "R7_the_day_set": day_sets_from_the_ledger(),
        "parameters": {
            "path": "live/pm_research/declarations/"
                    "de_multiday_gate1_params_v2.json",
            "supersedes": "live/pm_research/declarations/"
                          "de_multiday_gate1_params_v1.json",
            "digest_deliberately_NOT_carried_here": (
                "params v2 cites THIS declaration by digest. If this "
                "declaration also cited params by digest neither could ever "
                "be emitted -- each digest would depend on the other. The "
                "pin runs in ONE direction, params -> design, and this is "
                "the statement of which"),
            "what_v2_changed": [
                "run_not_before_utc split into read_not_before_utc + "
                "day_runs_allowed_for_closed_qualifying_days (R-572(B)(2))",
                "BE's cascade digest re-pointed at ab75b41, justified by a "
                "COMPUTED per-definition diff rather than by BE's commit "
                "message -- see params v2 `be_module_repoint`",
            ],
        },
        "R9_timing": TIMING_RULE,
        "R10_seal_layout": SEAL_LAYOUT,
        "dry_run_ledger_scope": DRY_RUN_LEDGER_SCOPE_DECLARED,
        "seed_convention": SEED_CONVENTION,
        "worktree_data_shell_trap": {
            "what": "in a seat worktree `<root>/data` is the "
                    "git-materialised shell holding only COMMITTED "
                    "artifacts; the real tree hangs off the R-397 symlink "
                    "at `<root>/data/data`",
            "how_it_bit_this_declaration": (
                "the R7 derivation read `<root>/data` and returned THREE "
                "qualifying days where the ledger has SIX -- 09-03, 09-04 "
                "and 09-05's verdicts were simply not in the shell. It was "
                "caught only because the count disagreed with a hand check "
                "made minutes earlier, not by any guard"),
            "the_fix": "the derivation resolves the symlink when it exists "
                       "and RECORDS the root it read, so a reader can tell "
                       "which tree produced the day set",
            "root_read_this_emission": None,
            "prior_instances": [
                "de_v2_owned_execution_input's 11th check FAILS in every "
                "seat worktree and PASSES in the main tree on "
                "byte-identical source (Q-DE-63)",
                "the reviewer hit the same trap from the other side this "
                "round -- its 'the ledger stops at 09-02' was a tracking "
                "gap, R-552"],
            "the_general_shape": (
                "a path that resolves in both trees but means different "
                "things in each. A digest cannot catch it, a green suite "
                "cannot catch it, and a count that nobody compares to a "
                "hand check cannot either"),
        },
        "R8_time_overrun": _r8_from_resources(),
        "_R8_removed_prose": {
            "withdrawn_text": "~20 hours of null across 2 arms x 5 days, "
                              "plus ~3 hours of replay",
            "why": "it sat beside `resources.totals`, which computes 11.26 "
                   "CPU-hours at G=5 -- a typed sentence contradicting the "
                   "computed field next to it. R8's estimate is now a "
                   "REFERENCE to those totals, so the two cannot diverge"},

        "what_DE_needs_from_BE_per_day": {
            "object": "the day's reference book -- the same shape as the "
                      "08-24 arms cache's `fr`: reference (slug -> side -> "
                      "generations with tranches), statuses, population, "
                      "n_slugs, and terminal_marks",
            "why_terminal_marks": "the inventory leg is valued to each "
                                  "window's terminal mark; a book without "
                                  "them reads NO_TERMINAL_MARK on every "
                                  "fill",
            "digest_pinning": (
                "BE publishes, per day, the book path and its sha256 in a "
                "builder declaration. DE's per-day artifact carries "
                "`reference_book: {day, path, sha256}` and RECOMPUTES the "
                "digest at read time; a mismatch REFUSES that day"),
            "de_does_not_build_the_book_and_does_not_open_BEs_pickle": True,
        },
        "decision_population": {
            "definition": "the arm's above-threshold events on day d at "
                          "the arm's FIXED theta -- the set a cancel "
                          "decision is drawn from",
            "on_the_consumed_hour_it_was": {
                "CONDVALUE_X_SKEW": {"decisions": 1154,
                                     "by_side": {"BUY_UP": 586,
                                                 "SELL_UP": 568}},
                "HAZARD_OVER_SKEWED_REF": {"decisions": 106,
                                           "by_side": {"BUY_UP": 41,
                                                       "SELL_UP": 65}}},
            "per_day_values_are_UNKNOWN_and_are_an_OUTPUT": True,
            "a_day_with_too_few_decisions_is_a_STATUS": (
                "if an arm's decision count on a day is 0, that day is "
                "reported as a counted status for that arm and the arm "
                "cannot be aggregated over G = 5 -- it does not silently "
                "become a 4-day test"),
        },
        "null": {
            "design": "random decisions matched to the arm's OWN count and "
                      "side split on that day, drawn from that day's "
                      "decision population, replayed through the SAME "
                      "stateful cascade",
            "matched_on": ["decision count", "side split"],
            "explicitly_not_matched_on": [
                "realised cancel count", "realised cancel set",
                "fills lost -- a decision is not a cancel"],
            "min_draws_per_arm_per_day": MIN_DRAWS,
            "machinery": CASCADE_MACHINERY,
            "draw_source": DRAW_SOURCE_RULE,
            "seed": {
                "rule": "seed = int(sha256(day_book_sha256 || arm || "
                        "'P003_GATE1_MULTIDAY')[:8], 16)",
                "why": "the seed PINS THE DATA: it is derived from the "
                       "day's book digest, so a book that moved cannot "
                       "reuse the same draw sequence, and the sequence is "
                       "reproducible from the artifact alone",
                "not_a_bare_integer": True,
                # v8: the prose rule above is unchanged and is no longer the
                # only statement of it. The FIELDS are what a reader and the
                # battery resolve.
                "convention": SEED_CONVENTION},
        },
        "metric": {
            "primary": "D(E0) -- net value delta at maker fee zero (our "
                       "signed rate), arm minus QR_SKEW_ONLY, per day",
            "robustness": "D(E-R) at the rebate's identity value, reported "
                          "beside it and never substituted for it",
            "unchanged_from": "R-537 / the fee-endpoint receipt v1-v3",
        },
        "per_day_location": {
            "statistic": "p = (1 + #{null >= observed}) / (1 + K), "
                         "ONE-SIDED, larger D is better",
            "floor_at_500_draws": 1 / (1 + MIN_DRAWS),
        },
        "day_cluster_aggregation": {
            "cluster_unit": "UTC day (rule 8)",
            "per_day_value": "Z_d = (D_d - mean(null_d)) / sd(null_d)",
            "estimate": "mean of Z_d over the G = 5 days",
            "interval": "reported ONLY because G >= 5; the exact sign test "
                        "over the 5 day signs is the test, and the "
                        "interval is the 5 Z values themselves -- no "
                        "normal approximation is claimed at n = 5",
            "why_not_pooled_over_draws": "pooling 2,500 draws across days "
                                         "would treat the DRAW as the "
                                         "cluster unit and inflate by the "
                                         "number of draws, which is free",
        },
        "section_7_predicate": {
            "statement": "arm a FAILS iff mean_d Z_(a,d) <= 0 OR the five "
                         "day signs are not unanimous",
            "declared_before_any_day_is_seen": True,
            "multiplicity_m": MULTIPLICITY,
            "alpha": ALPHA,
            "THE_ASYMMETRY_IS_THE_POINT": (
                "FAIL is cheap and needs no significance -- 'did not beat "
                "the null' is not a claim that needs power. PASS is capped "
                "by arithmetic: at G = 5 the smallest attainable one-sided "
                "sign-test p is 2^-5 = 0.03125, and Holm at m = 2 compares "
                "it against 0.025. NO ARM CAN CLEAR HOLM ON THIS RUN even "
                "if every day goes its way"),
            "so_a_pass_means": "DIRECTIONAL AND CONSISTENT, NEVER "
                               "SIGNIFICANCE-BEARING -- the same limit "
                               "R-529(A) ruled for the forward race, "
                               "declared here BEFORE the run rather than "
                               "discovered after it",
            "smallest_G_that_would_clear_holm": {
                "m_2": _smallest_G(ALPHA, 2), "m_1": _smallest_G(ALPHA, 1)},
            "and_that_is_ONE_MORE_DAY": (
                "at m = 2 the smallest clearing G is 6, so a SIXTH "
                "admissible day would make a unanimous pass "
                "significance-bearing. There is none yet -- 09-06 is the "
                "next candidate and is not complete -- but the price of a "
                "significant answer is one day, and that is worth knowing "
                "before the run rather than after it"),
            "this_is_stated_now_so_nobody_reads_a_pass_as_a_validation":
                True,
        },
        "falsifiers": {
            "a_planted_arm_that_MUST_FAIL": "an arm whose D sits at the "
                                            "null's median on every day",
            "a_planted_arm_that_MUST_PASS": "an arm above every draw on "
                                            "every day -- and the pass is "
                                            "asserted to be DIRECTIONAL, "
                                            "with clears_holm FALSE",
            "a_planted_day_with_a_WRONG_BOOK_DIGEST": "refuses that day "
                                                      "outright",
            "an_under_sampled_null": "fewer than 500 draws refuses",
            "a_degenerate_null": "zero dispersion is a STATUS, not a large "
                                 "Z",
            "a_missing_day": "G != 5 refuses rather than testing on 4",
        },
        "what_would_REFUTE_this_design": [
            "if the per-day decision population for an arm is systematically "
            "empty or tiny on the admissible days, the matched null cannot "
            "be built and Gate 1's control fails for a reason that is not "
            "about the arm -- that refutes the DESIGN, not the arm",
            "if BE's cascade cannot be driven on a day's book without "
            "re-fitting anything, the 'same cascade' premise is false",
            "if the five days' books are not independent in the way the "
            "day-cluster unit assumes (a single market event spanning "
            "days), the interval is wrong even at G = 5",
            "if the arm's theta is found to have been fitted on any of "
            "these five days, they are consumed and the run is void "
            "(rule 11)",
        ],
        "resources": _resources(),
        "what_this_declaration_is_not": {
            "a_run": False, "a_result": False,
            "it_touches_no_data": True,
            "gates_2_to_6": "if an arm passes, Gates 2-6 need FIVE FURTHER "
                            "admissible days (09-06 onward, earliest "
                            "complete set 09-10, readable 09-11) -- these "
                            "five are consumed by this run",
        },
    }


# --------------------------------------------------------------- selftest

LAST_BATTERY: dict = {}


def selftest(*, quiet: bool = False) -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            LAST_BATTERY.update({"outcome": "FAIL", "n_checks_run": n[0],
                                 "failed_on": label})
            raise SystemExit(f"[de_multiday_design_declaration] FAIL: {label}")
        n[0] += 1
        if not quiet:
            print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        try:
            fn()
        except DesignRefused as exc:
            if needle.lower() not in str(exc).lower():
                raise SystemExit(f"[de_multiday_design_declaration] FAIL: "
                                 f"{label} -- wrong reason: {exc}")
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_multiday_design_declaration] FAIL: {label} "
                         f"-- ADMITTED")

    d = declaration()
    # v8: THIS CHECK USED TO ASSERT THE STALE STATE. It required
    # `G_is_PENDING_the_USER_parameter is True` and so it PINNED the
    # declaration to a question the USER answered in R-555 four hours
    # earlier -- a check that held a stale field in place is worse than no
    # check, because it makes the staleness look verified.
    ok(d["days"]["G_is_PENDING_the_USER_parameter"] is False
       and d["days"]["STATUS"].startswith("RULED BY R-555")
       and d["days"]["v1_named"] and d["days"]["v1_G"] == 5,
       "v1's five named days stay recorded as provenance and the STATUS "
       "reads RULED BY R-555 -- the declaration states the ruled G rather "
       "than a pending question, and the check no longer pins the stale "
       "answer in place")
    _v6 = day_cluster_verdict({f"d{i}": 3.0 for i in range(6)}, g=6)
    ok(abs(_v6["p_one_sided_sign_test"] - 0.015625) < 1e-12
       and _v6["clears_holm"] is True
       and _v6["a_pass_is_significance_bearing"] is True,
       f"AND THE RULE IS DRIVEN AT G = 6, THE SET-A CASE: unanimous gives "
       f"2^-6 = {_v6['p_one_sided_sign_test']} which CLEARS Holm at m = 2 "
       f"-- so under SET A a pass IS significance-bearing, and the cap "
       f"that binds at G = 5 does not bind there")
    _v3 = day_cluster_verdict({f"d{i}": 3.0 for i in range(3)}, g=3)
    ok(_v3["clears_holm"] is False
       and _v3["p_one_sided_sign_test"] == 0.125,
       "and at G = 3, the SET-B case, a unanimous pass gives 0.125 and "
       "clears nothing -- the two sets are not a matter of degree")
    ok(d["theta"]["CONDVALUE_X_SKEW"] == 0.32450609461933483
       and d["theta"]["HAZARD_OVER_SKEWED_REF"] == 0.43525926488298716
       and all("sha256" in v for v in d["theta_pins"].values()),
       "both thetas are stated WITH the artifact and json path they are "
       "pinned at, so a reader can check they were not refitted")
    ok(d["null"]["min_draws_per_arm_per_day"] == 500
       and d["null"]["machinery"]["de_does_not_reimplement_it"] is True,
       "the null is >=500 draws per arm per day through BE's cascade, "
       "cited by path and NOT re-implemented here")
    ok("sha256(day_book_sha256" in d["null"]["seed"]["rule"],
       "the seed PINS THE DATA BY DIGEST -- derived from the day's book "
       "sha256, so a moved book cannot reuse the draw sequence")

    # ---- the rule, driven ----------------------------------------------
    null = [float(i) / 100 for i in range(500)]          # 0.00 .. 4.99
    loc = per_day_location(2.495, null)
    ok(loc["n_draws"] == 500 and abs(loc["floor"] - 1 / 501) < 1e-12
       and 0.0 < loc["p_one_sided"] < 1.0,
       f"per-day location is one-sided with floor 1/501 = "
       f"{loc['floor']:.6f}; a median arm reads p = "
       f"{loc['p_one_sided']:.4f}")
    refuses(lambda: per_day_location(1.0, null[:499]),
            "KNOWN-BAD: 499 draws is below the declared 500 and REFUSES",
            "below the declared minimum")
    refuses(lambda: per_day_standardised_excess(1.0, [3.0] * 500),
            "KNOWN-BAD: a null with zero dispersion is a STATUS, never a "
            "large Z", "zero dispersion")

    # A PLANTED ARM THAT MUST FAIL: sits at the null's mean every day.
    z_fail = {day: 0.0 for day in DAYS}
    v_fail = day_cluster_verdict(z_fail)
    ok(v_fail["FAILS_THE_SECTION_7_PREDICATE"] is True
       and v_fail["verdict"] == "FAILS_TO_BEAT_THE_REPLAY_NULL"
       and v_fail["signs_unanimous"] is False,
       "PLANTED ARM THAT MUST FAIL: an arm at the null's mean on every day "
       "FAILS -- mean Z is 0 and no day sign is positive")
    z_mixed = {DAYS[0]: 3.0, DAYS[1]: 3.0, DAYS[2]: 3.0, DAYS[3]: 3.0,
               DAYS[4]: -0.01}
    v_mixed = day_cluster_verdict(z_mixed)
    ok(v_mixed["FAILS_THE_SECTION_7_PREDICATE"] is True
       and v_mixed["mean_standardised_excess"] > 0
       and v_mixed["n_days_positive"] == 4,
       "AND A STRONG-BUT-NOT-UNANIMOUS ARM ALSO FAILS: mean Z is "
       "comfortably positive and ONE day disagrees. The stopping rule is "
       "declared on unanimity, not on the mean, so a four-of-five arm "
       "cannot be talked into a pass after the fact")

    # A PLANTED ARM THAT MUST PASS -- and the pass must be DIRECTIONAL.
    z_pass = {day: 4.0 for day in DAYS}
    v_pass = day_cluster_verdict(z_pass)
    ok(v_pass["FAILS_THE_SECTION_7_PREDICATE"] is False
       and v_pass["verdict"] == "BEATS_DIRECTIONALLY_NOT_SIGNIFICANTLY"
       and v_pass["signs_unanimous"] is True,
       "PLANTED ARM THAT MUST PASS: unanimous positive on all five days "
       "does NOT fail the section-7 predicate")
    ok(abs(v_pass["p_one_sided_sign_test"] - 0.03125) < 1e-12
       and abs(v_pass["holm_threshold_for_the_smaller_p"] - 0.025) < 1e-12
       and v_pass["clears_holm"] is False
       and v_pass["a_pass_is_significance_bearing"] is False,
       f"AND THE PASS IS CAPPED BY ARITHMETIC, DECLARED NOW: 2^-5 = "
       f"{v_pass['p_one_sided_sign_test']} against a Holm threshold of "
       f"{v_pass['holm_threshold_for_the_smaller_p']} at m = 2, so "
       f"clears_holm is FALSE even when every day goes the arm's way. A "
       f"pass is DIRECTIONAL, never significance-bearing")
    ok(_smallest_G(0.05, 2) == 6 and _smallest_G(0.05, 1) == 5
       and 2.0 ** -6 <= 0.025 and 2.0 ** -5 > 0.025
       and d["section_7_predicate"]["smallest_G_that_would_clear_holm"]
       == {"m_2": 6, "m_1": 5},
       "and the clearing G is COMPUTED, not asserted: 6 at m = 2 and 5 at "
       "m = 1. I first wrote 7 and 6 by hand and the check caught it -- "
       "so the price of a significance-bearing answer is ONE MORE DAY, "
       "not two, and that is known before anyone spends them")
    refuses(lambda: day_cluster_verdict({d: 1.0 for d in DAYS[:4]}),
            "KNOWN-BAD: four days REFUSE rather than quietly testing on a "
            "smaller G -- dropping a day after the fact is choosing after "
            "seeing", "exactly G = 5")

    # A PLANTED DAY WITH A WRONG BOOK DIGEST.
    ok(verify_book_digest("2026-09-01", "b.pkl", "a" * 64,
                          "a" * 64)["verified"] is True,
       "POSITIVE CONTROL, AND IT ADMITS: a day whose book digest matches "
       "its pin is verified")
    refuses(lambda: verify_book_digest("2026-09-01", "b.pkl", "a" * 64,
                                       "b" * 64),
            "PLANTED DAY WITH A WRONG BOOK DIGEST: the whole day REFUSES, "
            "not the offending draw", "digest mismatch")

    # ---- v2 items, each with a driven check where one is possible -----
    ok(d["R1_asm_the_scored_book"]["blocking_in_v1"] is True
       and "by_arm" in d["R1_asm_the_scored_book"][
           "what_asm_must_contain_per_day"]
       and "BOTH pinned heads" in d["R1_asm_the_scored_book"][
           "what_asm_must_contain_per_day"]["by_arm"],
       "R1: `asm` is declared with what it must contain per day -- by_arm "
       "keyed by (coin, head) for BOTH pinned heads, scored at the pinned "
       "thetas -- and its digest's route into the seed is stated")
    ok(d["R2_the_draw_pool"]["declared_choice"] == "SHARED"
       and str(HEAD_OVERLAP_FLOOR) in d["R2_the_draw_pool"]["refuses_if"],
       "R2: the draw pool is declared SHARED with the reason, the COST "
       "stated, and a numeric refusal bar (heads' scored sets overlapping "
       "by less than 0.90 of the smaller REFUSES the arm-day)")
    ok(d["R3_the_coin_set"]["declared"] == ["btc"]
       and "0fa2f1f7a5a4c58f" in d["R3_the_coin_set"]["why_btc_only"],
       "R3: btc only, because the PINNED THETAS ARE BTC THETAS -- and the "
       "eth siblings are named with their differing digests, so 'no eth "
       "model exists' is not claimed when the truth is that eth would be a "
       "different frozen object")

    # R4 driven, both directions.
    good = arm_day_admissible(200, [0.40 + 0.002 * i for i in range(500)])
    ok(good["admissible"] is True and good["status"] == "OK",
       f"R4 POSITIVE CONTROL, AND IT ADMITS: 200 decisions with a null "
       f"whose sd/|mean| is {good['sd_over_abs_mean']:.4f} is admissible")
    thin = arm_day_admissible(4, [0.40 + 0.001 * i for i in range(500)])
    ok(thin["admissible"] is False
       and "decisions 4" in thin["reasons"][0],
       "R4 KNOWN-BAD: four decisions REFUSE the arm-day -- the case v1 "
       "left to be decided after seeing it")
    tight = arm_day_admissible(200, [0.40 + 1e-6 * i for i in range(500)])
    ok(tight["admissible"] is False
       and any("sd" in r for r in tight["reasons"]),
       "R4 KNOWN-BAD, THE OTHER GAP: a SMALL-BUT-NONZERO sd refuses. v1's "
       "degenerate-null falsifier caught sd == 0 exactly and this is the "
       "case that makes Z explode")
    hazard_like = arm_day_admissible(106, None or [
        0.40033 + 0.16234 * ((i % 100) - 49.5) / 28.87 for i in range(500)])
    ok(hazard_like["sd_over_abs_mean"] > SD_FLOOR_FRACTION,
       f"R4 CALIBRATION, SO THE FLOOR CANNOT BE READ AS CHOSEN TO EXCLUDE "
       f"SOMETHING SEEN: a HAZARD-like null (mean 0.400, sd 0.162, ratio "
       f"0.4055) sits ABOVE the 0.25 floor -- the consumed hour would have "
       f"been admitted")

    ok(d["R5_the_smoke_is_sealed"]["all_G_days_run_regardless_of_interim_"
                                   "results"] is True
       and "D(E0)" in d["R5_the_smoke_is_sealed"]["day_1_does_NOT_publish"]
       and "absent from the artifact" in d["R5_the_smoke_is_sealed"][
           "artifact_level_refusal"],
       "R5: day 1 publishes RESOURCES ONLY, the economic fields are ABSENT "
       "from the artifact rather than present-and-ignored, and all G days "
       "run regardless of interim results")
    ok(d["R6_runtime_verification"]["on_mismatch"].startswith("REFUSE the "
                                                              "run"),
       "R6: theta and model digests are verified AT RUN TIME and a "
       "mismatch refuses THE RUN -- a moved model means the object under "
       "test is not the frozen one")
    r7 = d["R7_the_day_set"]
    # THE BLOCKER (reviewer efba2b6 §9): this asserted len == 6, G == 6 and
    # G == 3 against a ledger THAT GROWS BY CONSTRUCTION. At 00:06Z on
    # 09-07, when 09-06 is verdicted, all three would have failed -- the
    # date-dependent fixture rot DA fixed in round 54, reintroduced against
    # an input whose growth is SCHEDULED. Every count below is now an
    # OUTPUT and every assertion is a RELATION that holds whatever the
    # ledger says.
    qual = set(r7["qualifying_on_quality"])
    a_days = set(r7["SET_A_reads_count_as_untouched"]["days"])
    b_days = set(r7["SET_B_reads_consume_the_day"]["days"])
    ok(a_days == qual and b_days <= a_days
       and all(r7["day_read_state"][x]["previously_opened_for"]
               == OPENED_NONE for x in b_days)
       and all(r7["day_read_state"][x]["previously_opened_for"]
               != OPENED_NONE for x in a_days - b_days),
       f"R7 IS A RELATION, NOT A COUNT: SET A is exactly the qualifying "
       f"set, SET B is a SUBSET of it, every day in B reads "
       f"`previously_opened_for == none` and every day in A-but-not-B does "
       f"not. Counts are OUTPUTS -- today {len(qual)} qualify -- and this "
       f"assertion survives the ledger growing at 00:06Z tomorrow")
    for lbl, blk in (("SET_A", r7["SET_A_reads_count_as_untouched"]),
                     ("SET_B", r7["SET_B_reads_consume_the_day"])):
        h = blk["holm"]
        ok(h["G"] == len(blk["days"])
           and h["clears_holm_at_m2"] == (2.0 ** -h["G"] <= ALPHA / 2)
           and h["clears_holm_at_m1"] == (2.0 ** -h["G"] <= ALPHA),
           f"and {lbl}'s Holm outcome is COMPUTED FROM ITS OWN G "
           f"({h['G']}), not asserted: clears at m=2 iff 2^-G <= "
           f"{ALPHA / 2}. The arithmetic is checked; the value of G is "
           f"whatever the ledger yields")
    ruled_closed = [x for x in RULED_DAY_SET
                    if r7["ledger_rows_as_read"].get(x, {}).get(
                        "day_closed_calendar") is True]
    missing = [x for x in ruled_closed if x not in qual]
    # FALSIFIER A: PLANT AN EXTRA QUALIFYING DAY -- the ledger growing at
    # 00:06Z tomorrow is exactly this, and it MUST STILL PASS.
    grown = json.loads(json.dumps(r7))
    grown["qualifying_on_quality"] = r7["qualifying_on_quality"] + ["2026-09-06"]
    grown["SET_A_reads_count_as_untouched"]["days"] = list(
        grown["qualifying_on_quality"])
    grown["SET_B_reads_consume_the_day"]["days"] = (
        r7["SET_B_reads_consume_the_day"]["days"] + ["2026-09-06"])
    gq = set(grown["qualifying_on_quality"])
    ga = set(grown["SET_A_reads_count_as_untouched"]["days"])
    gb = set(grown["SET_B_reads_consume_the_day"]["days"])
    ok(ga == gq and gb <= ga
       and all(DAY_READ_STATE[x]["previously_opened_for"] == OPENED_NONE
               for x in gb),
       f"FALSIFIER A -- THE LEDGER GROWS AND THE ASSERTION STILL PASSES: "
       f"planting 2026-09-06 as a seventh qualifying day (which is what "
       f"00:06Z on 09-07 will do) leaves every relation true. The old "
       f"assertion asserted 6, 6 and 3 and would have gone RED on "
       f"schedule")
    # FALSIFIER B: REMOVE A RULED DAY FROM THE QUALIFYING SET -- must refuse.
    shrunk = [x for x in r7["qualifying_on_quality"] if x != "2026-09-03"]
    ok("2026-09-03" in RULED_DAY_SET
       and r7["ledger_rows_as_read"]["2026-09-03"]["day_closed_calendar"]
       and "2026-09-03" not in shrunk,
       "FALSIFIER B -- A RULED, CLOSED DAY MISSING FROM THE QUALIFYING SET "
       "is exactly the condition the next assertion refuses on: 09-03 is "
       "ruled, closed, and would be absent")
    ok(bool([x for x in RULED_DAY_SET
             if r7["ledger_rows_as_read"].get(x, {}).get(
                 "day_closed_calendar") is True and x not in shrunk]),
       "and the refusal condition FIRES on that shrunken set -- the check "
       "is not one that can only ever pass")

    ok(not missing,
       f"AND EVERY RULED DAY THAT IS CLOSED QUALIFIES: "
       f"{len(ruled_closed)} of the ruled set are closed and all of them "
       f"pass the four conjuncts and day-quality. A ruled day that closed "
       f"and did NOT qualify is the condition R-555 says waits for the "
       f"next qualifying closed day, and it must be visible here")
    ok("QUALITY is the bar" in r7["version_is_NOT_a_bar"]
       and "2026-08-29" in r7["qualifying_on_quality"],
       "and v1's imported bar is WITHDRAWN: R-497(F)(1) says version is "
       "not a bar, QUALITY is -- which admits 08-29, the day v1 excluded "
       "for a reason the USER never set")
    # ---- (1) the withdrawn sentence, and the per-day field ------------
    r7d = d["R7_the_day_set"]
    ok(r7d["day_read_state"]["2026-09-01"]["previously_opened_for"]
       == OPENED_INTERIM
       and r7d["day_read_state"]["2026-09-03"]["previously_opened_for"]
       == OPENED_NONE
       and r7d["SET_B_reads_consume_the_day"]["derived_from_the_field"]
       is True,
       "(1) the per-day read state is a FIELD -- 09-01/09-02 "
       "interim_read_of_frozen_candidate, 08-29 development_read, "
       "09-03..05 none -- and SET B is DERIVED FROM IT rather than from a "
       "membership test against prose")
    ok(r7d["withdrawn_sentence"]["withdrawn_by"].startswith("R-549(A)")
       and "sealed and unread" in r7d["withdrawn_sentence"]["text"]
       and "NOTHING ABOUT THE ARMS" in r7d["withdrawn_sentence"][
           "what_survives_of_it"],
       "and the withdrawn sentence is carried ONLY as the thing being "
       "withdrawn, with its surviving half stated separately -- the two "
       "halves travelled together in v1 and v2 and that is how the false "
       "one survived")
    _dsr = d["days"]
    ok("IS WITHDRAWN" in _dsr[
           "nothing_about_the_ARMS_has_been_chosen_on_them"],
       "and the `days` block no longer ASSERTS the withdrawn claim; it "
       "names it as withdrawn and points at the field")
    ok(d["R8_time_overrun"]["estimate_source"].startswith(
        "resources.totals")
       and abs(d["R8_time_overrun"]["sequential_cpu_hours"]["G5"]
               - d["resources"]["totals"]["G5"]["sequential_cpu_hours"])
       < 1e-12
       and "20 hours" in d["_R8_removed_prose"]["withdrawn_text"],
       "(3) R8's estimate is a COMPUTED REFERENCE to resources.totals, "
       "equal to it by construction; the withdrawn '~20 hours' sentence is "
       "carried as removed prose with the reason -- it sat beside a "
       "computed field that said 11.26")
    _fake = {"a": {"b": "the race scored it sealed and unread"}}
    try:
        _withdrawn_phrase_audit(_fake)
        ok(False, "KNOWN-BAD: the withdrawn phrase in an UNDECLARED "
                  "context was admitted")
    except DesignRefused as _e:
        ok("outside a declared withdrawal context" in str(_e),
           "KNOWN-BAD: the withdrawn phrase appearing anywhere but a "
           "declared withdrawal context REFUSES the emission -- so a grep "
           "hit at the artifact is answerable by a field instead of by "
           "reading three sentences")
    # ---- (1) R2's floor, CALIBRATED -----------------------------------
    fc = d["R2_the_draw_pool"]["floor_calibration"]
    ok(fc["observed_overlap"] == 1.0
       and fc["measurement"]["intersection"] == 29813
       and fc["floor_admits_the_consumed_hour"] is True,
       f"(1) R2's floor is CALIBRATED on the consumed hour exactly as "
       f"R4's was: both heads score the SAME 29,813 generations, overlap "
       f"{fc['observed_overlap']} (Jaccard 1.0), so the "
       f"{fc['floor']} floor ADMITS the precedent run -- and the "
       f"shared-pool choice costs EXACTLY NOTHING there, because there is "
       f"only one pool to share")
    ok(HEAD_OVERLAP_FLOOR < 1.0
       and not (1.0000001 <= HEAD_OVERLAP_OBSERVED),
       "FALSIFIER: a floor ABOVE the observed overlap would refuse the "
       "consumed hour itself -- 1.0000001 > 1.0 excludes it. And because "
       "the observed value is the CEILING of the statistic, no floor "
       "below 1.0 can have been chosen to clear it")
    # ---- (2) R5 cites the code path and the field that proves it fired -
    cp = d["R5_the_smoke_is_sealed"]["IT_IS_NOW_A_CODE_PATH_NOT_A_PROMISE"]
    ok(cp["emitter"] == "de_multiday_gate1_runner.seal()"
       and "assert_no_economic_leak" in cp["guard"]
       and "caught the emitter on its first run" in cp["proof_it_fired"]
       and cp["fixture_receipt_field"].startswith(
           "per_day_sealed_artifacts"),
       "(2) R5 names the emitter, the guard, the ONE shared name list, "
       "and the fixture-receipt field that proves the guard fired -- it "
       "is a code path now, not a promise the runner must keep")
    # ---- (4) the root derivation REFUSES ------------------------------
    r7x = d["R7_the_day_set"]
    ok(r7x["ledger_root_resolved"] == DECLARED_LEDGER_ROOT
       and r7x["root_verified_at_run_time"] is True,
       f"(4) the ledger root is RESOLVED through the symlink and compared "
       f"to the declared {DECLARED_LEDGER_ROOT} at run time")
    import tempfile as _tf2
    with _tf2.TemporaryDirectory() as _td3:
        (Path(_td3) / "data/pm_5min/derived").mkdir(parents=True)
        try:
            day_sets_from_the_ledger(_td3)
            ok(False, "(4) KNOWN-BAD: a PLANTED SHELL ROOT returned a day "
                      "set instead of refusing")
        except DesignRefused as _e:
            ok("not the declared" in str(_e) and _td3 in str(_e),
               "(4) KNOWN-BAD, A PLANTED SHELL ROOT: the derivation "
               "REFUSES and NAMES THE ROOT. It used to return an empty day "
               "set silently -- an answer that looks like a result while "
               "looking at the wrong tree")

    ok(d["worktree_data_shell_trap"]["how_it_bit_this_declaration"]
       .startswith("the R7 derivation read")
       and len(d["worktree_data_shell_trap"]["prior_instances"]) == 2,
       "and the worktree data-shell trap is recorded as a field with how "
       "it bit THIS declaration and its two prior instances -- a path that "
       "resolves in both trees and means different things in each")

    r = d["resources"]
    ok(abs(r["per_day_BOTH_arms"]["null_hours"] - 1.9393) < 1e-3
       and abs(r["totals"]["G5"]["sequential_cpu_hours"] - 11.26) < 0.02
       and abs(r["totals"]["G6"]["sequential_cpu_hours"] - 13.52) < 0.02
       and "factor of two" in r["v1_mislabelled_this"],
       f"ADDENDUM 2: the resource arithmetic is COMPUTED from the measured "
       f"seconds -- 290.9 s is one hour for BOTH arms, so it is "
       f"{r['per_day_BOTH_arms']['null_hours']:.3f} h of null per DAY, "
       f"{r['totals']['G5']['sequential_cpu_hours']:.2f} sequential CPU-h "
       f"at G=5 and {r['totals']['G6']['sequential_cpu_hours']:.2f} at "
       f"G=6. v1 overstated the null by a factor of two")
    ok(r["mandatory_wrapper"].startswith("flock -n")
       and "MemoryMax=8G" in r["mandatory_wrapper"]
       and "CPUQuota=100%" in r["mandatory_wrapper"]
       and r["mandatory_first_run"].startswith("the ONE-DAY SMOKE"),
       "ADDENDUM 3 (SEAT_PROTOCOL rule 20): the heavy-run wrapper is named "
       "verbatim -- flock refuses if another heavy run holds the lock -- "
       "and the ONE-DAY SMOKE with sealed economic fields is declared the "
       "mandatory first run")
    ok(d["R8_time_overrun"]["the_500_draw_minimum_is_protected_by"]
       == "REFUSING THE ARM-DAY"
       and "lowering the draw count" in d["R8_time_overrun"]["never_by"],
       "R8: a time overrun is answered by REFUSING THE ARM-DAY, never by "
       "cutting draws or raising the cap -- written down in advance "
       "because cutting draws is the tempting response")

    ok(d["section_7_predicate"]["multiplicity_m"] == 2
       and d["section_7_predicate"][
           "this_is_stated_now_so_nobody_reads_a_pass_as_a_validation"],
       "multiplicity m = 2 is declared in the document, not inferred at "
       "read time")
    ok(len(d["what_would_REFUTE_this_design"]) >= 4,
       f"the declaration says what would REFUTE IT, not only what would "
       f"refute the arms: {len(d['what_would_REFUTE_this_design'])} "
       f"conditions")
    ok(d["resources"]["cap"].startswith("one CPU, MemoryMax=8G")
       and "day REFUSES rather than the cap rising"
       in d["resources"]["cap"],
       "the resource declaration names the cap AND what happens when a "
       "day exceeds it -- the day refuses, the cap never rises (R-174)")
    ok(d["decision_population"]["per_day_values_are_UNKNOWN_and_are_an_"
                                "OUTPUT"] is True,
       "the per-day decision counts are declared as OUTPUTS, so no "
       "expectation about them can be quietly turned into a filter")

    # ================= v8: THE CONVENTIONS THAT LIVED ONLY IN CODE ========
    import de_multiday_gate1_runner as _RUN

    # ---- the seed convention, driven against the runner's own expression --
    _cases = [("0" * 64, "CONDVALUE_X_SKEW"), ("f" * 64, "HAZARD_OVER_SKEWED_REF"),
              ("a1b2c3" + "0" * 58, "CONDVALUE_X_SKEW")]
    _agree = [(seed_from_convention(b, a), _RUN.seed_for(b, a))
              for b, a in _cases]
    ok(all(x == y for x, y in _agree) and len({x for x, _ in _agree}) == 3,
       f"THE DECLARED SEED CONVENTION IS THE ONE THE RUNNER IMPLEMENTS -- a "
       f"reference built from the DECLARED FIELDS ALONE reproduces "
       f"seed_for() on {len(_cases)} distinct (book, arm) pairs, and the "
       f"three seeds are distinct so the check is not passing on a constant: "
       f"{[x for x, _ in _agree]}")

    # ---- INTERIOR CONTROL, one mutation per declared field ---------------
    _muts = {
        "domain_separator": {"domain_separator": "P003_GATE1_MULTIDAY_X"},
        "joiner": {"joiner": "-"},
        "hex_truncation_chars": {"hex_truncation_chars": 16},
        "int_base": {"int_base": 32},
        "field_order": {"field_order": ["arm", "day_book_sha256",
                                        "domain_separator"]},
        "encoding": {"encoding": "utf-16"},
    }
    _still_equal = []
    for _f, _patch in _muts.items():
        _c = dict(SEED_CONVENTION)
        _c.update(_patch)
        if seed_from_convention("7" * 64, "CONDVALUE_X_SKEW", _c) == \
                _RUN.seed_for("7" * 64, "CONDVALUE_X_SKEW"):
            _still_equal.append(_f)
    ok(not _still_equal,
       f"AND EVERY DECLARED FIELD IS LOAD-BEARING -- mutating any ONE of "
       f"{sorted(_muts)} changes the seed. A convention with an inert field "
       f"is a convention that did not need declaring, and a check that "
       f"cannot notice the mutation is rule 16's control that cannot fail "
       f"(fields that survived mutation: {_still_equal or 'none'})")

    ok(SEED_CONVENTION["hex_truncation_chars"] == 8
       and SEED_CONVENTION["domain_separator"] == "P003_GATE1_MULTIDAY"
       and "[:8], 16" in SEED_CONVENTION["expression"],
       "the two values R-572(B)(4) names by hand -- the literal "
       "`P003_GATE1_MULTIDAY` and the 8-hex truncation -- are the values in "
       "the declared fields, so the register entry and the artifact agree "
       "without a reader translating between them")

    # ---- the --dry-run-ledger scope, declared vs the runner's own words --
    _rd = _RUN.dry_run_scope_as_the_runner_states_it()
    ok(_rd["reads"] == DRY_RUN_LEDGER_SCOPE_DECLARED["reads"]
       and _rd["does_NOT_read"] ==
       DRY_RUN_LEDGER_SCOPE_DECLARED["does_NOT_read"],
       f"THE DECLARED --dry-run-ledger SCOPE IS THE SCOPE THE RUNNER STATES "
       f"-- two independently authored lists compared, not one constant read "
       f"twice: reads {_rd['reads']}")
    ok(set(DRY_RUN_LEDGER_SCOPE_DECLARED["does_NOT_verify"]) ==
       {"P2_BE_reference_book", "P6_pinned_models_and_thetas",
        "P7_BE_cascade_module_digest"}
       and not any(f in str(_rd["reads"]) for f in ("book", "model", "theta",
                                                    "cascade")),
       "AND THE SCOPE NAMES THE PLAYBOOK PRECONDITIONS IT DOES NOT COVER -- "
       "P2, P6 and P7. A green dry run exits 0 and prints a receipt while "
       "BE's book, the pinned models and BE's cascade digest are entirely "
       "unexamined; that gap is invisible at the console and is a field now")

    # ---- R9 / R10, the two rulings as fields -----------------------------
    ok(TIMING_RULE["day_runs_allowed_for_closed_qualifying_days"] is True
       and TIMING_RULE["read_not_before_utc"] == "2026-09-09T00:06:00Z"
       and "AGGREGATE READ only" in TIMING_RULE["what_that_date_governs"],
       "R9: the not-before date governs the AGGREGATE READ and per-day "
       "sealed runs are allowed on closed qualifying days (R-572(B)(2)) -- "
       "the field params v1 carried as one whole-set `run_not_before_utc` "
       "is split into the two things it was doing")
    ok(set(SEAL_LAYOUT["keys_present_in_BOTH_states"]) ==
       {"sealed", "seal_status", "sealed_at_every_depth",
        "sealed_field_names", "economic"}
       and SEAL_LAYOUT["unsealed_values"]["sealed_at_every_depth"] is False,
       "R10: the five seal-layout keys are declared present in BOTH states "
       "with explicit values, and the one that is absent by design "
       "(`economic`, when sealed) is declared as a RULE rather than left to "
       "a consumer to discover as a None")

    # ---- the version axis ------------------------------------------------
    ok(PROTOCOL.endswith(f"_V{VERSION}")
       and DECLARATION_CHAIN[-1][0].startswith(
           f"p003_de_multiday_gate1_design_v{VERSION - 1}__")
       and len(DECLARATION_CHAIN) == VERSION - 1,
       f"THE VERSION TRAVELS IN ONE PLACE: protocol {PROTOCOL} ends in "
       f"V{VERSION}, the chain holds {len(DECLARATION_CHAIN)} = VERSION - 1 "
       f"predecessors and its head is v{VERSION - 1}. v7 on disk read "
       f"protocol V4, filename v7 and supersedes v2")
    _sup = d["supersedes"]
    ok(_sup["path"].endswith(DECLARATION_CHAIN[-1][0])
       and _sup["sha256"] == DECLARATION_CHAIN[-1][1]
       and len(_sup["chain"]) == len(DECLARATION_CHAIN),
       "and `supersedes.path` resolves to the IMMEDIATE predecessor -- the "
       "field an automated reader follows (rule 13). v7's named v2 and skipped "
       "four versions")
    _chain = verify_declaration_chain()
    ok(_chain["n_entries"] == len(DECLARATION_CHAIN)
       and _chain["every_digest_recomputed"] is True
       and all(r["agrees"] for r in _chain["entries"]),
       f"AND EVERY CHAIN DIGEST IS READ FROM ITS FILE, not trusted as typed "
       f"-- {_chain['n_entries']} recomputed and equal. This round typed one "
       f"digest that was INVENTED and one that was the literal "
       f"'PLACEHOLDER'; computing them is what caught both")
    _bad_chain = list(DECLARATION_CHAIN[:-1]) + [
        (DECLARATION_CHAIN[-1][0], "0" * 64)]
    _saw = None
    try:
        verify_declaration_chain(chain=_bad_chain)
    except DesignRefused as exc:
        _saw = str(exc)
    ok(_saw is not None and "does not match the artifacts" in _saw,
       "KNOWN-BAD, THE OTHER DIRECTION: a chain entry whose typed digest is "
       "not the file's REFUSES -- so the check above is one that can fail")

    ok(d["days"]["G_is_PENDING_the_USER_parameter"] is False
       and d["days"]["G_RULED"] == 6
       and d["days"]["ruled_set"] == ["2026-09-03", "2026-09-04",
                                      "2026-09-05", "2026-09-06",
                                      "2026-09-07", "2026-09-08"]
       and d["days"]["ruled_set"] == _RUN.load_params()["days"],
       "AND R-555 IS IN THE DECLARATION, not only in the parameter file: "
       "G_RULED = 6 on the ruled set, `G_is_PENDING_the_USER_parameter` is "
       "FALSE -- it read True for four hours after the USER answered, and "
       "the CLI summary was printing that stale value -- and the ruled set "
       "here EQUALS the one the runner loads, so the two cannot drift")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    LAST_BATTERY.update({
        "outcome": "PASS",
        "n_checks_run": n[0],
        "expected_checks_in_the_source": EXPECTED_CHECKS,
        "run_count_equals_source_expected": n[0] == EXPECTED_CHECKS,
        "ran_in_the_emitting_process": True,
        "v1_defect_this_closes": (
            "v1's receipt recorded n_checks 19 while the source's "
            "EXPECTED_CHECKS was 20 -- main() wrote EXPECTED_CHECKS - 1 "
            "instead of the count the run produced. The source hash "
            "matched, so it was a RECEIPT-COUNT defect and not a code "
            "difference. Both numbers are now carried WITH their computed "
            "equality, so a receipt cannot disagree with its own source "
            "again"),
    })
    if not quiet:
        print(f"[de_multiday_design_declaration] PASS -- {n[0]} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.emit or a.output is None:
        ap.error("choose --selftest or --emit --output PATH")
    me = Path(__file__).resolve()
    payload = declaration()
    payload["as_of"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat()
    payload["source_identity"] = {
        "producing_code": me.name,
        "producing_code_sha256": hashlib.sha256(me.read_bytes()).hexdigest(),
        **carrying_commit_block(me),
    }
    LAST_BATTERY.clear()
    selftest(quiet=True)
    payload["battery"] = dict(LAST_BATTERY)
    payload["data_root"] = DR.require_canonical(
        "the multi-day design declaration")
    payload["worktree_data_shell_trap"]["root_read_this_emission"] = \
        payload["R7_the_day_set"]["ledger_root_read"]
    payload["withdrawn_phrase_audit"] = _withdrawn_phrase_audit(payload)
    if a.output.exists():
        raise DesignRefused(f"output already exists: {a.output}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    r7 = payload["R7_the_day_set"]
    print(json.dumps({
        "emitted": str(a.output), "status": payload["status"],
        # NOT the module constant, and no longer "PENDING" either: the
        # CLI printed `G: 5` from `G`, then printed PENDING for four hours
        # after R-555 answered. It now prints the payload's OWN field.
        "G": payload["days"]["G_RULED"],
        "G_source": "the emitted payload's days.G_RULED",
        "SET_A_G": r7["SET_A_reads_count_as_untouched"]["holm"]["G"],
        "SET_B_G": r7["SET_B_reads_consume_the_day"]["holm"]["G"],
        "arms": list(ARMS),
        "battery": payload["battery"]["outcome"],
        "battery_checks": payload["battery"]["n_checks_run"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
