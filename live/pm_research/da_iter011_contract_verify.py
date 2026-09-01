#!/usr/bin/env python3
"""DA's INDEPENDENT contract-level verifier for the iteration-011 artifact.

Separate implementation on purpose (R-235: do-not-harmonize). This reader
shares no code with `phase2_iter011*.py` -- it reads the emitted JSON and
recomputes every arithmetic claim from the artifact's own fields.

WHAT IT REFUSES TO DO. It does not re-run the fit, does not import BE's
generators, and does not read a conclusion string as evidence: every claim
below is a computed predicate (rule 10).

THE READ MUST PROVE ITSELF (R-289's matched pair). Two vacuums were filed one
hour apart in this programme: one matched ZERO cells (caught by an implausible
count), one matched ALL cells and read NONE of their fields (caught only by the
other seat's disclosure). A count assertion alone would have caught the first
and missed the second. So this reader carries BOTH:

  * a POPULATION counter -- how many cells were actually visited; and
  * a TYPED-FIELD counter -- how many times each field was read AS THE TYPE
    the checks filter it as.

`assert_read()` REFUSES when either is zero for a field a check depends on, so
"found nothing" can never be returned by a reader that touched nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "pm_5min" / "derived"
ARTIFACT = DERIVED / "iter011_conditional_value_v1__coin_btc.json"

# The 12-file code lattice, transcribed from `phase2_arms.CODE_IDENTITY_FILES`
# and re-derived here rather than imported: importing BE's module to check
# BE's identity would make the check depend on the thing it is checking.
CODE_IDENTITY_FILES = (
    "phase2_arms.py", "phase2_declaration.py", "phase2_embargo.py",
    "phase2_state_schema_freeze.py", "harmful_action_eval.py",
    "harmful_hazard_model.py", "harmful_fast_compute.py",
    "harmful_state_features.py", "harmful_exposure_rows.py",
    "flow_intensity.py", "flow_fill_development.py",
    "harmful_candidate_manifest.py",
)

# R-306, verbatim from the register: "Q3's two slope gates compose as
# CONJUNCTION + WORSE SIDE (both CIs must exclude 0; the cell's p = the worse
# -- the Q2-min logic; family stays 24)."
R306_CONJUNCTS = ("both_slopes_present", "worse_side_statistic",
                  "p_is_worse_of_the_two", "both_CIs_exclude_zero")


class Reads:
    """Population and typed-field counters. The instrument's own falsifier."""

    def __init__(self) -> None:
        self.cells = 0
        self.fields: dict[str, int] = {}
        self.type_errors: list[str] = []

    def typed(self, obj: dict, key: str, types, ctx: str,
              allow_missing: bool = False):
        """Read `obj[key]` AS `types`, counting the read. A wrong type is a
        recorded error, never a silent default -- the default is what turned
        twelve read cells into twelve zeros."""
        if key not in obj:
            if allow_missing:
                self.fields.setdefault(f"{key}:absent", 0)
                self.fields[f"{key}:absent"] += 1
                return None
            self.type_errors.append(f"{ctx}: field {key!r} ABSENT")
            return None
        v = obj[key]
        if v is None and allow_missing:
            self.fields.setdefault(f"{key}:null", 0)
            self.fields[f"{key}:null"] += 1
            return None
        if not isinstance(v, types):
            self.type_errors.append(
                f"{ctx}: field {key!r} is {type(v).__name__}, "
                f"not {getattr(types, '__name__', types)}")
            return None
        self.fields.setdefault(key, 0)
        self.fields[key] += 1
        return v

    def assert_read(self, required: dict[str, int]) -> None:
        """REFUSE unless the population and every depended-on field were read.

        This is the whole point of the class: a check that filtered on a field
        nobody successfully read returns 'nothing found' from an empty set."""
        if self.cells == 0:
            raise SystemExit(
                "REFUSED: 0 cells read. A verification claim must assert its "
                "parse actually reached the population (R-289).")
        missing = {k: (self.fields.get(k, 0), n) for k, n in required.items()
                   if self.fields.get(k, 0) < n}
        if missing:
            raise SystemExit(
                "REFUSED: fields not read as typed -- "
                + "; ".join(f"{k}: read {got} of {want}"
                            for k, (got, want) in sorted(missing.items()))
                + ". A filter over a field the parse never read is vacuous "
                  "(R-289, the field-level half).")
        if self.type_errors:
            raise SystemExit("REFUSED: type errors -- "
                             + "; ".join(self.type_errors[:8]))


def _sha16(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def combined_identity(files: dict[str, str | None]) -> str:
    """`phase2_arms.measured_code_identity`'s formula, re-implemented."""
    return hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(files.items())).encode()
    ).hexdigest()[:16]


def check(out: list, name: str, ok: bool, detail: str, evidence: Any = None):
    out.append({"check": name, "pass": bool(ok), "detail": detail,
                "evidence": evidence})


def verify(doc: dict, tree_identity: bool = True) -> dict:
    r = Reads()
    res: list[dict] = []

    fam = r.typed(doc, "family", dict, "root") or {}
    decl = r.typed(doc, "declared_family", dict, "root") or {}
    cells = r.typed(fam, "cells", dict, "family") or {}
    results = r.typed(doc, "results", dict, "root") or {}

    # ---- C1/C2 the 24 declared cells, and the grid that defines them -------
    arms = r.typed(decl, "arms", list, "declared_family") or []
    heads = r.typed(decl, "heads", list, "declared_family") or []
    budgets = r.typed(decl, "budgets", list, "declared_family") or []
    grid = {f"{a}/{h}/{b}" for a in arms for h in heads for b in budgets}
    n_cells_decl = r.typed(decl, "n_cells", int, "declared_family")
    declared_list = set(r.typed(decl, "cells", list, "declared_family") or [])

    per_cell: dict[str, dict] = {}
    for key in sorted(cells):
        c = cells[key]
        if not isinstance(c, dict):
            r.type_errors.append(f"cell {key}: not a dict")
            continue
        r.cells += 1
        per_cell[key] = {
            "status": r.typed(c, "status", str, key),
            "p_value": r.typed(c, "p_value", (int, float), key,
                               allow_missing=True),
            "holm_p": r.typed(c, "holm_p", (int, float), key,
                              allow_missing=True),
            "n_actions": r.typed(c, "n_actions", int, key),
            "statistic": r.typed(c, "statistic", (int, float), key),
            "head": r.typed(c, "head", str, key),
            "arm": r.typed(c, "arm", str, key),
            "survives": r.typed(c, "survives_joint_reading_at_0_05", bool, key),
            "detail": r.typed(c, "detail", str, key),
        }

    check(res, "declared_family_is_24_cells",
          n_cells_decl == 24 and len(grid) == 24 and len(per_cell) == 24
          and declared_list == grid == set(per_cell),
          "arms x heads x budgets == declared list == emitted cells",
          {"n_cells_declared": n_cells_decl, "grid": len(grid),
           "emitted": len(per_cell),
           "missing_from_emission": sorted(grid - set(per_cell)),
           "extra_in_emission": sorted(set(per_cell) - grid)})

    # ---- C3 unevaluable cells occupy their slots as STATUSES (A1.4) --------
    without_p = {k for k, v in per_cell.items() if v["p_value"] is None}
    with_p = set(per_cell) - without_p
    by_status: dict[str, int] = {}
    for v in per_cell.values():
        by_status[v["status"]] = by_status.get(v["status"], 0) + 1
    stated_by_status = r.typed(fam, "cells_by_status", dict, "family") or {}
    n_with = r.typed(fam, "n_cells_with_p_value", int, "family")
    n_without = r.typed(fam, "n_cells_without_p_value", int, "family")
    check(res, "unevaluable_cells_occupy_their_slots",
          len(without_p) + len(with_p) == 24
          and sum(by_status.values()) == 24
          and by_status == stated_by_status
          and n_with == len(with_p) and n_without == len(without_p)
          and all(per_cell[k]["status"] != "OK" for k in without_p),
          "no cell is dropped; every p-less cell carries a non-OK status and "
          "the status tally covers all 24",
          {"recomputed_by_status": by_status, "stated": stated_by_status,
           "n_with_p": len(with_p), "n_without_p": len(without_p)})

    hd = r.typed(fam, "holm_denominator", int, "family")
    hd_declared = r.typed(fam, "holm_denominator_is_declared_not_evaluated",
                          bool, "family")
    check(res, "holm_denominator_is_the_DECLARED_24",
          hd == 24 and hd_declared is True and hd == n_cells_decl,
          "Holm divides by the declared family size, not by the evaluated "
          "subset -- unevaluable cells cannot be dropped to make the "
          "correction smaller",
          {"holm_denominator": hd, "declared_not_evaluated": hd_declared})

    # ---- C4 every cell carries its n and its population --------------------
    pops = r.typed(doc, "populations", dict, "root") or {}
    coin = r.typed(doc.get("COIN_SLICE", {}), "coin", str, "COIN_SLICE")
    n_ok = all(isinstance(v["n_actions"], int) and v["n_actions"] > 0
               for v in per_cell.values())
    check(res, "every_cell_carries_n_and_a_named_population",
          n_ok and coin in pops and isinstance(pops.get(coin), dict),
          "each cell states n_actions and the artifact names the population "
          "it was computed on",
          {"all_cells_have_positive_n": n_ok, "coin": coin,
           "population_keys": sorted(pops.get(coin, {}))})

    # The n a cell CARRIES vs the n its statistic was computed ON.
    head_n: dict[tuple[str, str], int] = {}
    for arm, ablock in (results.get(coin) or {}).items():
        for hname, h in (ablock.get("heads") or {}).items():
            if isinstance(h, dict) and isinstance(h.get("n_actions"), int):
                head_n[(arm, hname)] = h["n_actions"]
                r.fields["head_n_actions"] = r.fields.get("head_n_actions", 0) + 1
    HEAD_SOURCE = {"Q1_arrival": ["Q1_arrival"],
                   "Q2_sign": ["Q2_p_pos", "Q2_p_neg"],
                   "Q3_magnitudes": ["Q3_m_good", "Q3_m_harm"],
                   "Q4_combined_ev": []}
    mism = []
    for k, v in sorted(per_cell.items()):
        srcs = [head_n.get((v["arm"], s)) for s in HEAD_SOURCE.get(v["head"], [])]
        srcs = [s for s in srcs if s is not None]
        if srcs and v["n_actions"] not in srcs:
            mism.append({"cell": k, "cell_n_actions": v["n_actions"],
                         "head_n_actions": srcs})
    check(res, "cell_n_equals_the_n_its_statistic_was_computed_on",
          not mism,
          "a cell quoting an arrival-population n beside a conditional-head "
          "statistic overstates the population behind the number (rule 8: "
          "every quoted population carries ITS n)",
          {"n_mismatched_cells": len(mism), "examples": mism[:4]})

    # ---- C5 permutation arithmetic ----------------------------------------
    draws, floors, obs_p = set(), set(), set()
    for arm, ablock in (results.get(coin) or {}).items():
        for hname, h in (ablock.get("heads") or {}).items():
            mr = h.get("matched_random") if isinstance(h, dict) else None
            if not isinstance(mr, dict):
                continue
            nd = r.typed(mr, "n_draws", int, f"{arm}/{hname}/matched_random")
            pv = r.typed(mr, "p_value", (int, float),
                         f"{arm}/{hname}/matched_random")
            if nd is not None:
                draws.add(nd)
                floors.add(1.0 / (nd + 1))
            if pv is not None:
                obs_p.add(pv)
    floor = 1.0 / 501.0
    check(res, "permutation_floor_is_1_over_n_draws_plus_1",
          draws == {500} and floors == {floor} and obs_p == {floor},
          "500 draws => the smallest attainable p is 1/501; every evaluable "
          "head sits exactly ON that floor (0 draws beat the observed)",
          {"n_draws_values": sorted(draws), "floor": floor,
           "observed_p_values": sorted(obs_p),
           "min_draws_bar_rule6": 200, "meets_rule6_minimum": min(draws or [0]) >= 200})

    holm_bad = [{"cell": k, "p": v["p_value"], "holm_p": v["holm_p"],
                 "expected": None if v["p_value"] is None else 24 * v["p_value"]}
                for k, v in sorted(per_cell.items())
                if (v["p_value"] is None) != (v["holm_p"] is None)
                or (v["p_value"] is not None
                    and v["holm_p"] != 24 * v["p_value"])]
    check(res, "holm_p_equals_declared_denominator_times_cell_p",
          not holm_bad,
          "holm_p == 24 * p in every evaluable cell, and is null exactly where "
          "p is null",
          {"n_bad": len(holm_bad), "examples": holm_bad[:3]})

    at_floor = 24 * floor
    next_step = 24 * (2.0 / 501.0)
    check(res, "family_clears_0_05_ONLY_at_the_permutation_floor",
          at_floor < 0.05 and next_step > 0.05,
          "at the floor the adjusted p is 0.0479 (<0.05). ONE draw beating the "
          "observed in a cell moves that cell to 2/501: 0.0958 under the flat "
          "x24 this artifact applies, or x13 = 0.0518 under a true step-down "
          "with the other eleven still tied. BOTH exceed 0.05, so the family's "
          "survival is one permutation draw wide either way",
          {"holm_at_floor": at_floor, "holm_at_next_attainable_p_flat24": next_step,
           "holm_at_next_attainable_p_stepdown13": 13 * (2.0 / 501.0),
           "bar": 0.05})

    # Holm vs Bonferroni are indistinguishable here BECAUSE the p are tied.
    tied = len({v["p_value"] for v in per_cell.values()
                if v["p_value"] is not None}) == 1
    check(res, "holm_step_down_is_DISTINGUISHABLE_from_bonferroni",
          not tied,
          "with every evaluable p tied at the floor, Holm's step-down and a "
          "flat Bonferroni x24 give identical numbers (monotonicity carries "
          "the first step across the ties). The artifact therefore cannot "
          "evidence WHICH procedure ran; it would separate them only on "
          "untied p",
          {"all_evaluable_p_tied": tied,
           "distinct_evaluable_p": sorted({v["p_value"] for v in per_cell.values()
                                           if v["p_value"] is not None})})

    # ---- C6 the Q4 contradiction, read at the artifact ----------------------
    ina = r.typed(doc, "incumbent_null_applicability", dict, "root") or {}
    comparable = r.typed(ina, "comparable", dict,
                         "incumbent_null_applicability") or {}
    na_heads = r.typed(ina, "not_applicable_heads", list,
                       "incumbent_null_applicability") or []
    status_for_those = r.typed(ina, "status_for_those_cells", str,
                               "incumbent_null_applicability")
    q4_comparable = r.typed(comparable, "Q4_combined_ev", bool, "comparable")
    q4_cells = {k: v for k, v in per_cell.items() if v["head"] == "Q4_combined_ev"}
    q4_status = {v["status"] for v in q4_cells.values()}
    econ_none, econ_unpaired = 0, 0
    for arm, ablock in (results.get(coin) or {}).items():
        for b, e in (ablock.get("economics") or {}).items():
            if not isinstance(e, dict):
                continue
            if r.typed(e, "incumbent_net_cents", (int, float),
                       f"{arm}/econ/{b}", allow_missing=True) is None:
                econ_none += 1
            if r.typed(e, "paired_against_incumbent", bool,
                       f"{arm}/econ/{b}") is False:
                econ_unpaired += 1
    q4_detail_says_own = all("not an increment" in (v["detail"] or "")
                             for v in q4_cells.values())
    contradiction = (q4_comparable is True and q4_status == {"NO_INCUMBENT_COUNTERPART"}
                     and econ_none == 6 and econ_unpaired == 6
                     and q4_detail_says_own)
    check(res, "Q4_declaration_agrees_with_Q4_cells",
          not contradiction,
          "incumbent_null_applicability declares Q4 comparable:true, while all "
          "six Q4 cells carry NO_INCUMBENT_COUNTERPART, every economics block "
          "has incumbent_net_cents=null and paired_against_incumbent=false, "
          "and each detail states the net is the CANDIDATE'S OWN value and not "
          "an increment. Four independent fields, one direction",
          {"comparable.Q4_combined_ev": q4_comparable,
           "q4_cell_statuses": sorted(q4_status),
           "economics_blocks_with_null_incumbent": econ_none,
           "economics_blocks_unpaired": econ_unpaired,
           "all_q4_details_say_not_an_increment": q4_detail_says_own,
           "contradiction_confirmed_in_artifact": contradiction})

    # The declaration's own handling promise, applied to the OTHER na head.
    q3_cells = {k: v for k, v in per_cell.items() if v["head"] == "Q3_magnitudes"}
    q3_status = {v["status"] for v in q3_cells.values()}
    check(res, "declared_status_reaches_every_not_applicable_head",
          all(q3_status == {status_for_those} for _ in [0]) if "Q3_magnitudes" in na_heads else True,
          "the declaration says the affected cells REPORT "
          f"{status_for_those!r}; Q3_magnitudes is named in "
          "not_applicable_heads but its six cells carry a different status, so "
          "the incumbent gap is not carried as a status there at all",
          {"not_applicable_heads": na_heads,
           "status_for_those_cells": status_for_those,
           "q3_cell_statuses": sorted(q3_status),
           "q2_cell_statuses": sorted({v["status"] for v in per_cell.values()
                                       if v["head"] == "Q2_sign"})})

    # ---- C7 the survivor predicate ----------------------------------------
    surv_flagged = {k for k, v in per_cell.items() if v["survives"]}
    surv_listed = set(r.typed(fam, "surviving_cells", list, "family") or [])
    surv_non_ok = sorted(k for k in surv_flagged if per_cell[k]["status"] != "OK")
    check(res, "survivor_predicate_conjuncts_status_with_holm",
          not surv_non_ok,
          "cells marked survives_joint_reading_at_0_05 while carrying a non-OK "
          "status: survival is derived from the Holm comparison ALONE, so a "
          "cell with no incumbent counterpart is published as a survivor",
          {"n_flagged": len(surv_flagged),
           "flagged_equals_listed": surv_flagged == surv_listed,
           "survivors_with_non_OK_status": surv_non_ok,
           "their_statuses": sorted({per_cell[k]["status"] for k in surv_non_ok})})

    # ---- C8 is the preserved Q3 evidence SUFFICIENT for R-306? -------------
    q3 = {}
    for arm, ablock in (results.get(coin) or {}).items():
        hs = ablock.get("heads") or {}
        side = {}
        for s in ("Q3_m_good", "Q3_m_harm"):
            h = hs.get(s)
            if isinstance(h, dict):
                side[s] = {
                    "calibration_slope": r.typed(h, "calibration_slope",
                                                 (int, float), f"{arm}/{s}"),
                    "p": (h.get("matched_random") or {}).get("p_value"),
                    "n_actions": h.get("n_actions"),
                    "ci": next((k for k in h if "ci" in k.lower()
                                or "interval" in k.lower()), None),
                }
        q3[arm] = side
    both_slopes = all(len(v) == 2 and all(x["calibration_slope"] is not None
                                          for x in v.values())
                      for v in q3.values())
    both_p = all(all(x["p"] is not None for x in v.values()) for v in q3.values())
    any_ci = any(x["ci"] for v in q3.values() for x in v.values())
    clus = r.typed(doc, "cluster_disclosure", dict, "root") or {}
    intervals_claimable = r.typed(clus, "intervals_claimable", bool,
                                  "cluster_disclosure")
    G = r.typed(clus, "G_complete_utc_days", int, "cluster_disclosure")
    check(res, "R306_conjunct_both_slopes_present", both_slopes,
          "per-side calibration slopes preserved for both arms",
          {a: {s: v["calibration_slope"] for s, v in d.items()}
           for a, d in q3.items()})
    check(res, "R306_conjunct_p_is_worse_of_the_two", both_p,
          "per-side matched-random p preserved for both arms; the worse of the "
          "two is computable without a refit (here both sit on the floor, so "
          "the worse IS the floor)",
          {a: {s: v["p"] for s, v in d.items()} for a, d in q3.items()})
    check(res, "R306_conjunct_both_CIs_exclude_zero_is_EVALUABLE",
          bool(any_ci),
          "R-306 requires BOTH slope CIs to exclude 0. The artifact carries no "
          "interval of any kind for either slope, and its own cluster "
          "disclosure states intervals are not claimable at G=0 complete UTC "
          "days (rule 8). So this conjunct is not merely missing from the "
          "emission -- it is unobtainable on this population, and a refit "
          "would not supply it",
          {"any_ci_field_found": bool(any_ci),
           "intervals_claimable": intervals_claimable,
           "G_complete_utc_days": G,
           "ruled_unit": clus.get("ruled_unit"),
           "unit_used": clus.get("unit_used"),
           "matched_random_p_is_not_a_CI": True})

    worse = {}
    for arm, d in q3.items():
        slopes = {s: v["calibration_slope"] for s, v in d.items()}
        cell_stat = next((v["statistic"] for k, v in per_cell.items()
                          if v["arm"] == arm and v["head"] == "Q3_magnitudes"),
                         None)
        min_slope = min(slopes.values()) if slopes else None
        min_dev = (min(slopes, key=lambda s: abs(slopes[s] - 1.0))
                   if slopes else None)
        worse[arm] = {"slopes": slopes, "cell_statistic": cell_stat,
                      "min_slope": min_slope,
                      "min_abs_deviation_side": min_dev,
                      "min_abs_deviation_value": slopes.get(min_dev)}
    check(res, "Q3_cell_statistic_is_the_WORSE_side",
          all(v["cell_statistic"] == v["min_slope"] for v in worse.values()),
          "the emitted Q3 statistic equals min(slope_good, slope_harm) -- the "
          "worse side under a CI-excludes-0 gate, which is what R-306 rules",
          worse)
    rule_strings = {arm: (ablock.get("adjudicated_statistics") or {}).get("Q3_cell_rule")
                    for arm, ablock in (results.get(coin) or {}).items()}
    describes = all(v["cell_statistic"] == v["min_abs_deviation_value"]
                    for v in worse.values())
    check(res, "Q3_rule_STRING_describes_the_computation_performed",
          describes,
          "the carried rule string says 'min |calibration slope deviation| "
          "side reported', which selects the side CLOSEST to 1 -- the BETTER "
          "side. The number emitted is min(slope), the worse side. They "
          "coincide on this data only because both slopes are below 1; the "
          "string would misdescribe the computation on any data where one "
          "slope exceeds 1 (rule 10: a conclusion beside a computed number)",
          {"rule_strings": rule_strings,
           "computed_is_min_slope": all(v["cell_statistic"] == v["min_slope"]
                                        for v in worse.values()),
           "computed_is_min_abs_deviation": describes})

    # ---- C9 identity, recomputed with the code's own formula ---------------
    ident = r.typed(doc, "identity", dict, "root") or {}
    files = r.typed(ident, "fit_code_files", dict, "identity") or {}
    stated = r.typed(ident, "fit_code_sha256_prefix", str, "identity")
    recomputed = combined_identity(files) if files else None
    check(res, "identity_is_self_consistent",
          recomputed == stated and set(files) == set(CODE_IDENTITY_FILES),
          "the combined lattice hash recomputed from the artifact's OWN "
          "per-file map, using the formula in phase2_arms.measured_code_"
          "identity, equals the value it declares -- over exactly the 12 "
          "lattice files",
          {"stated": stated, "recomputed_from_artifact": recomputed,
           "n_files": len(files),
           "files_match_lattice": set(files) == set(CODE_IDENTITY_FILES)})

    if tree_identity:
        live = {n: _sha16(ROOT / "live" / "pm_research" / n)
                for n in CODE_IDENTITY_FILES}
        moved = sorted(n for n in CODE_IDENTITY_FILES if live[n] != files.get(n))
        check(res, "identity_still_matches_the_working_tree",
              not moved,
              "re-hashing the 12 lattice files on disk reproduces the "
              "artifact's per-file map, so the code that produced this "
              "artifact is the code now in the tree",
              {"combined_now": combined_identity(live),
               "combined_in_artifact": stated, "files_moved": moved})

    ref = r.typed(ident, "fit_code_ref", str, "identity", allow_missing=True)
    check(res, "fit_code_ref_is_recorded",
          bool(ref),
          "R-306's standing rule requires a result-bearing run to record the "
          "COMMITTED producer. The content hash is present and verifiable; the "
          "commit ref field is null, so the artifact cannot name the commit it "
          "ran from",
          {"fit_code_ref": ref,
           "content_hash_present": bool(stated)})

    # ---- C10 data artifacts: byte counts against disk ----------------------
    byte_rows = []
    for pk, bk in (("tape_path", "tape_bytes"), ("fragment_path", "fragment_bytes"),
                   ("topup_path", "topup_bytes")):
        p = ident.get(pk)
        want = r.typed(ident, bk, int, "identity")
        got = os.stat(p).st_size if p and os.path.exists(p) else None
        byte_rows.append({"path": p, "declared": want, "on_disk": got,
                          "match": want == got})
    check(res, "declared_data_artifact_byte_counts_match_disk",
          all(x["match"] for x in byte_rows),
          "the tape/fragment/topup byte counts the artifact binds to are the "
          "byte counts on disk now (content shas NOT re-hashed here -- 5.1 GB; "
          "stated so the scope of this check is not overread)",
          byte_rows)

    # ---- C11 the evidence class -------------------------------------------
    dev = r.typed(doc, "development_evidence", dict, "root") or {}
    is_val = r.typed(dev, "is_a_validation", bool, "development_evidence")
    check(res, "artifact_declares_itself_development_not_validation",
          is_val is False and intervals_claimable is False and G == 0,
          "development evidence, computed from the population label and the "
          "complete-day count rather than asserted",
          {"is_a_validation": is_val, "G_complete_utc_days": G,
           "intervals_claimable": intervals_claimable})

    # ---- C12 an as-of, which rule 8 requires beside every quoted n ---------
    def _find_asof(o, path="") -> list[str]:
        hits = []
        if isinstance(o, dict):
            for k, v in o.items():
                if any(t in k.lower() for t in ("as_of", "asof", "generated_at",
                                                "run_started", "utc_ts")):
                    hits.append(f"{path}.{k}")
                hits += _find_asof(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                hits += _find_asof(v, f"{path}[{i}]")
        return hits
    asof = _find_asof(doc)
    check(res, "artifact_carries_an_as_of",
          bool(asof),
          "rule 8: every quoted population carries its n AND its as-of. The "
          "n are present throughout; no as-of / generated-at timestamp exists "
          "anywhere in the artifact",
          {"as_of_fields_found": asof})

    # The reader must prove it read the population AND the fields it filtered.
    r.assert_read({"status": 24, "n_actions": 24, "statistic": 24,
                   "survives_joint_reading_at_0_05": 24, "head": 24,
                   "arm": 24, "detail": 24, "p_value": 12,
                   "n_draws": 10, "calibration_slope": 4,
                   "paired_against_incumbent": 6, "cells": 1,
                   "comparable": 1, "fit_code_files": 1})

    return {"artifact": str(ARTIFACT), "checks": res,
            "n_checks": len(res),
            "n_failing": sum(1 for c in res if not c["pass"]),
            "reads": {"cells": r.cells, "fields": dict(sorted(r.fields.items()))}}


# --------------------------------------------------------------------------
# falsifiers: a known-bad it must refuse, and a positive control it must admit
# --------------------------------------------------------------------------
def _selftests(doc: dict) -> int:
    import copy
    checks = 0
    fails = 0

    def ok(cond, label):
        nonlocal checks, fails
        checks += 1
        if not cond:
            fails += 1
            print(f"FAIL: {label}")

    def named(rep, name):
        return next(c for c in rep["checks"] if c["check"] == name)

    real = verify(doc, tree_identity=False)

    # POSITIVE CONTROL (rule 16: a control that only ever refuses passes
    # nothing). The structural contract checks must ADMIT the real artifact.
    for n in ("declared_family_is_24_cells", "unevaluable_cells_occupy_their_slots",
              "holm_denominator_is_the_DECLARED_24",
              "permutation_floor_is_1_over_n_draws_plus_1",
              "holm_p_equals_declared_denominator_times_cell_p",
              "family_clears_0_05_ONLY_at_the_permutation_floor",
              "identity_is_self_consistent",
              "every_cell_carries_n_and_a_named_population"):
        ok(named(real, n)["pass"], f"positive control: {n} must ADMIT the real artifact")
    ok(real["reads"]["cells"] == 24, "positive control: 24 cells actually read")

    # KNOWN-BAD 1: a dropped cell must be caught, not tolerated.
    d1 = copy.deepcopy(doc)
    d1["family"]["cells"].pop("composed_lgbm/Q1_arrival/5%")
    try:
        r1 = verify(d1, tree_identity=False)
        ok(not named(r1, "declared_family_is_24_cells")["pass"],
           "known-bad: a dropped cell must fail the 24-cell check")
    except SystemExit:
        ok(True, "known-bad: dropped cell refused")

    # KNOWN-BAD 2 (the CELL-level vacuum, MEM's half of R-289): zero cells.
    d2 = copy.deepcopy(doc)
    d2["family"]["cells"] = {}
    try:
        verify(d2, tree_identity=False)
        ok(False, "known-bad: an empty cell set must REFUSE, not report clean")
    except SystemExit as e:
        ok("0 cells read" in str(e), "known-bad: empty cell set refused by count")

    # KNOWN-BAD 3 (the FIELD-level vacuum, the coordinator's half of R-289):
    # every cell present, the filtered field renamed. A count assertion alone
    # passes this; only the typed-read counter catches it.
    d3 = copy.deepcopy(doc)
    for c in d3["family"]["cells"].values():
        c["p"] = c.pop("p_value")
    try:
        verify(d3, tree_identity=False)
        ok(False, "known-bad: renamed p_value must REFUSE, not read as 0 cells with p")
    except SystemExit as e:
        ok("not read as typed" in str(e) or "ABSENT" in str(e),
           "known-bad: field-level vacuum refused")

    # KNOWN-BAD 4: broken Holm arithmetic must be caught.
    d4 = copy.deepcopy(doc)
    d4["family"]["cells"]["composed_lgbm/Q1_arrival/5%"]["holm_p"] = 0.001
    r4 = verify(d4, tree_identity=False)
    ok(not named(r4, "holm_p_equals_declared_denominator_times_cell_p")["pass"],
       "known-bad: a wrong holm_p must fail")

    # KNOWN-BAD 5: the denominator shrunk to the EVALUATED subset.
    d5 = copy.deepcopy(doc)
    d5["family"]["holm_denominator"] = 12
    r5 = verify(d5, tree_identity=False)
    ok(not named(r5, "holm_denominator_is_the_DECLARED_24")["pass"],
       "known-bad: a 12-cell denominator must fail the declared-denominator rule")

    # DISCRIMINATION CONTROLS -- these prove the three findings are READ off
    # the artifact rather than hardcoded. Flip the source field; the finding
    # must go away.
    d6 = copy.deepcopy(doc)
    d6["incumbent_null_applicability"]["comparable"]["Q4_combined_ev"] = False
    r6 = verify(d6, tree_identity=False)
    ok(named(r6, "Q4_declaration_agrees_with_Q4_cells")["pass"],
       "discrimination: with comparable:false the Q4 contradiction must CLEAR")
    ok(not named(real, "Q4_declaration_agrees_with_Q4_cells")["pass"],
       "discrimination: it must FIRE on the real artifact")

    d7 = copy.deepcopy(doc)
    for k, c in d7["family"]["cells"].items():
        if c.get("survives_joint_reading_at_0_05"):
            c["status"] = "OK"
    r7 = verify(d7, tree_identity=False)
    ok(named(r7, "survivor_predicate_conjuncts_status_with_holm")["pass"],
       "discrimination: all-OK survivors must CLEAR the survivor check")
    ok(not named(real, "survivor_predicate_conjuncts_status_with_holm")["pass"],
       "discrimination: it must FIRE on the real artifact")

    d8 = copy.deepcopy(doc)
    for arm in d8["results"]["btc"].values():
        for s in ("Q3_m_good", "Q3_m_harm"):
            arm["heads"][s]["calibration_slope_ci"] = [0.4, 0.9]
    r8 = verify(d8, tree_identity=False)
    ok(named(r8, "R306_conjunct_both_CIs_exclude_zero_is_EVALUABLE")["pass"],
       "discrimination: an added CI field must make the CI conjunct evaluable")
    ok(not named(real, "R306_conjunct_both_CIs_exclude_zero_is_EVALUABLE")["pass"],
       "discrimination: with no CI anywhere it must FIRE on the real artifact")

    d9 = copy.deepcopy(doc)
    d9["identity"]["fit_code_files"]["phase2_arms.py"] = "0" * 16
    r9 = verify(d9, tree_identity=False)
    ok(not named(r9, "identity_is_self_consistent")["pass"],
       "known-bad: a mutated per-file sha must break the combined identity")

    print(f"selftest: {checks - fails}/{checks} passed")
    return 1 if fails else 0


def main() -> int:
    global ARTIFACT
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        # A subject the caller NAMES, so the same reader can be pointed at a
        # superseding artifact without a second implementation drifting from
        # this one. The subject is echoed in the report; a check that does not
        # say what it read is not a check.
        ARTIFACT = Path(args[0]).resolve()
    doc = json.loads(ARTIFACT.read_text())
    if "--selftest" in sys.argv:
        return _selftests(doc)
    # A red selftest must not be able to produce numbers.
    if _selftests(doc) != 0:
        print("REFUSING to report: selftest RED")
        return 2
    rep = verify(doc)
    print(json.dumps(rep, indent=1, sort_keys=False))
    print(f"\n{rep['n_checks'] - rep['n_failing']}/{rep['n_checks']} contract "
          f"checks hold; {rep['n_failing']} FAIL")
    print(f"reads: {rep['reads']['cells']} cells; "
          f"{sum(rep['reads']['fields'].values())} typed field reads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
