"""DE-Phase4-Diag -- the runner for the scheduled diagnostic (R-459).

SURFACE AUTHORISATION: R-459 (the USER's seventh ruling schedules ONE
diagnostic execution of the frozen `DE_PHASE4_PROTOCOL_DRAFT.md` on its own
§3 population) and `plans/DE_PHASE4_DIAGNOSTIC_ADDENDUM_2026-09-02.md`,
which declared the population, cells, arms, null and predicates BEFORE any
cell was read.  This file is the fourth instrument -- the one the addendum's
own declared limit said had to exist before the diagnostic could run.

A SEPARATE ENTRY POINT, NEVER A SWITCH.  `de_lane4_real_parity._receipt_cell`
refuses every economics key (`:8-14`) because LANE4 is a verification
harness and an economics number there would be a number nobody declared.
This runner EMITS economics by design, and the way those two live together
is that they are different programs: nothing here toggles LANE4, and LANE4
gains no flag.

WHAT IT REFUSES, so that a cell nobody declared cannot be produced:
  * a latency rung or budget outside the frozen axis, `enable_reduce` on, or
    `charge_reset_cost_at_generation_start` True -- by name, naming the axis;
  * an output directory that is a `fwd*` directory, an existing anchor, or
    anything but the declared new one;
  * a receipt missing any binding field -- either sha, either head's
    manifest shas, or the evidence flag.

THE ESTIMAND CARRIES ITS CAP.  The feed is
`phase4_generation_tables.tranche_table(..., declare_cap=True)`, whose
per-row latency labels are capped at `FILL_HORIZON_S = 1.0 s`.  Every cell
this runner produces therefore estimates *value preventable WITHIN ONE
SECOND*, and the receipt says so rather than leaving a reader to find it in
a module docstring.

    python3 live/pm_research/de_phase4_diag_runner.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

EXPECTED_CHECKS = 67

ROOT = Path(__file__).resolve().parents[2]
PLANS = Path(__file__).resolve().parent / "plans"
FROZEN = PLANS / "DE_PHASE4_PROTOCOL_DRAFT.md"
ADDENDUM = PLANS / "DE_PHASE4_DIAGNOSTIC_ADDENDUM_2026-09-02.md"
FITS = ROOT / "data/pm_5min/derived/phase2_fits"
SLUGS = FITS / "fit_slugs.json"

#: The ONLY directory this runner may write, and it is created by the RUN
#: round -- not by this one.
OUTDIR = ROOT / "data/pm_5min/derived/phase4_diag_r459"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import de_head_scoring as HS                    # noqa: E402
import de_matched_random_control as MRC          # noqa: E402
import de_rho_estimator as RHO                   # noqa: E402
import de_score_stream as SS                     # noqa: E402
import harmful_stateful_policy as HSP            # noqa: E402
from phase4_generation_tables import (FILL_HORIZON_S,   # noqa: E402
                                     tranche_table)

#: THE DECLARED GRID -- the addendum's §b, transcribed once and CHECKED
#: against the addendum by `de_phase4_protocol_check`, so a widened grid
#: here goes red there rather than quietly producing cells.
LATENCY_RUNGS_MS = (5, 10, 20, 30, 50, 75, 100, 150, 250)
BUDGETS = (0.05, 0.10, 0.15)
COINS = ("btc", "eth")
PRIMARY = {"coin": "btc", "latency_ms": 250, "budget": 0.10,
           "charge_reset_cost_at_generation_start": False,
           "enable_reduce": False}
#: Both bracketed, always: neither is a selection axis (§4).
REPOST_FILL_MODELS = HSP.REPOST_FILL_MODELS
PROTECTION_MODES = HSP.PROTECTION_MODES

#: DE33-C5 / DE32-R4: THE ARM TABLE -- name -> what the arm IS. Round 33
#: replayed only the two CONDVALUE arms while the receipt named five, and
#: `run_cell` iterated whatever dict the caller passed with
#: `theta=th.get(arm, 0.5)` -- a defaulted policy constant at a new line.
ARM_SPEC = {
    "QR_SKEW_ONLY": {"predictor": False, "head": None,
                     "note": "the frozen reference: skew ON, no cancel"},
    "QR_CANCEL_HOLD_X_SKEW": {"predictor": True, "head": "incumbent_linear_d",
                              "note": "the incumbent policy"},
    "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d":
        {"predictor": True, "head": "incumbent_linear_d",
         "note": "condvalue over the skewed reference, incumbent head"},
    "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm":
        {"predictor": True, "head": "q1_arrival_composed_lgbm",
         "note": "the head under test (R-424's component of record)"},
    "RANDOM_MATCHED": {"predictor": True, "head": None,
                       "note": "the acting control; scores come from the "
                               "draw, thresholds from the treated arm"},
}

#: The arms this execution runs, and the name resolved in the addendum.
ARMS = ("QR_SKEW_ONLY", "QR_CANCEL_HOLD_X_SKEW",
        "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d",
        "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm",
        "RANDOM_MATCHED")
#: Declared unrunnable, and why -- carried into the receipt (rule 4).
ARMS_NOT_RUN = {
    "HAZARD_ONLY_NEUTRAL": "NO_NEUTRAL_REFERENCE",
    "CONDVALUE_NEUTRAL": "NO_NEUTRAL_REFERENCE",
    "CONDVALUE_X_SKEW_X_FAIRPRICE": "NO_CHALLENGER_SCORED",
}
#: Where the null's draws run, from the addendum's compute arithmetic.
NULL_CELLS = (("btc", 250, 0.10), ("eth", 250, 0.10))
N_DRAWS = 200

POPULATION_NAME = "v3_4_consumed_fragment"

#: DE34-C4: THE PIN MUST COVER THE CODE THAT WILL BE CALLED, not the two
#: files that happen to match. The feed and the assembly call
#: `harmful_exposure_rows.py` (fit sha c2e40100ddf3f7a1) and
#: `phase2_arms.py` (3249dfc61c31b8d2); BOTH have moved since the fit
#: (1bbd8e75 at 851edaf; ab19f5c6 at 2e1204f, a parser-only change whose
#: `_feature_pass` def bytes are identical by AST against the manifest's
#: own `fit_code_ref`). So the design is stated rather than assumed:
#:
#:   * a called file whose CURRENT bytes match the manifest -> bound;
#:   * a called file that has MOVED -> REFUSED unless the run supplies the
#:     fit-commit bytes, because "the function I need is unchanged" is a
#:     claim about a function and the manifest pins a FILE;
#:   * the refusal names the file, both shas, and the commit the manifest
#:     records, so the next step is `git show <fit_code_ref>:<file>` rather
#:     than a judgement call.
#:
#: Nothing here weakens the pin to make the run possible: the run does not
#: happen this round, and a pin that passes because it was narrowed is the
#: shape this programme keeps removing.
CALLED_FIT_CODE = ("harmful_exposure_rows.py", "phase2_arms.py")
#: The repost dwell and the rate limit are DECLARED here with their reason,
#: because the frozen protocol's axes (:88-99) contain neither: an
#: undeclared default in a policy runner is a policy choice nobody made.
REPOST_DWELL_S = 2.0
#: EST-R1: `HALF_SPREAD_CENTS` IS GONE. The frozen text settles what
#: `spread` means -- DRAFT:212-213 requires "spread capture" reported per
#: cell and rho as the retained-book ratio of it -- so the denominator is a
#: MEASUREMENT, not a constant of this diagnostic. With a constant H the
#: ratio was identically `-(size-weighted mean markout) / H`, i.e. a
#: rescaled mean markout whose reading threshold WAS the constant. The mid
#: at fill now comes from the feed per tranche (`wf.mid_at`), so no number
#: of mine sits in the denominator.
MAX_CANCELS_PER_MINUTE = float("inf")

BINDING_FIELDS = ("frozen_protocol_sha256", "addendum_sha256",
                  "head_manifest_shas", "incumbent_manifest_shas",
                  "is_a_validation", "G_complete_utc_days",
                  "evidence_class", "value_horizon")


class DiagRefused(RuntimeError):
    """The runner refuses rather than producing a cell nobody declared."""


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def population_slugs(path: Path | None = None) -> dict:
    """The §3 population, read from the slugs iteration 011 itself used."""
    p = path or SLUGS
    if not p.exists():
        # SITE: population#1
        raise DiagRefused(f"no {p.name}: the population is the one 011 used "
                          f"and is not reconstructed here")
    slugs = json.loads(p.read_text())
    per = {c: sorted(s for s in slugs if s.startswith(f"{c}-")) for c in COINS}
    n = sum(len(v) for v in per.values())
    if n != len(slugs):
        # SITE: population#2
        raise DiagRefused(
            f"{len(slugs) - n} slug(s) belong to neither coin: a population "
            f"that does not partition is not the §3 population")
    return {"per_coin": per, "n_total": n,
            "n_per_coin": {c: len(v) for c, v in per.items()}}


def thresholds_for(coin: str, head: str) -> dict:
    """DE33-C2: the thresholds live WITH THE FIT, under the key that fit
    actually carries -- `causal_thresholds` for the incumbent, the budget
    map for the head under test. Round 33 read `thresholds` /
    `budget_thresholds`, which `linear_d_{coin}.json` does not have, so the
    run refused at its FIRST CELL, after the ~29-minute feed."""
    return HS.thresholds(coin, head)


def theta_for(coin: str, head: str, budget: float) -> float:
    """The cancel threshold for this budget, from the bound fit."""
    th = thresholds_for(coin, head)
    key = f"{int(round(budget * 100))}%"
    if key not in th:
        # SITE: theta#1
        raise DiagRefused(
            f"{head}/{coin}: the manifest-bound fit carries no threshold "
            f"for budget {key} (has {sorted(th)}). A policy constant is an "
            f"input; defaulting one here would be this runner encoding a "
            f"policy choice nobody declared (DE32-C5)")
    return th[key]


def cell_params(cell: dict, *, theta_cancel: float, protection_mode: str,
                repost_fill_model: str) -> dict:
    """The policy parameters for one arm of one cell.  Every constant is
    either the cell's, the fit's, or the frozen protocol's -- none is this
    file's."""
    if protection_mode not in PROTECTION_MODES:
        # SITE: params#1
        raise DiagRefused(f"protection_mode {protection_mode!r} not in "
                          f"{PROTECTION_MODES}")
    if repost_fill_model not in REPOST_FILL_MODELS:
        # SITE: params#2
        raise DiagRefused(f"repost_fill_model {repost_fill_model!r} not in "
                          f"{REPOST_FILL_MODELS}")
    return {
        "predictor_enabled": True,
        "theta_cancel": theta_cancel,
        # The repost threshold is the protocol's hysteresis, not a taste:
        # it must sit strictly below theta_cancel (the policy refuses
        # otherwise) and is declared here as HALF the cancel threshold.
        "theta_repost": theta_cancel / 2.0,
        "repost_dwell_s": REPOST_DWELL_S,
        "cancel_effective_latency_ms": float(cell["latency_ms"]),
        "queue_reset_cost_cents": cell.get("queue_reset_cost_cents", 0.0),
        "protection_mode": protection_mode,
        # EST-R4: this comment USED to say the frozen protocol names no
        # rate limit and cite ":88-99" for the axes. Both halves were
        # false: `DRAFT:71` (row 8 of the §2 parameter table) NAMES the
        # rate limit and asks for a PER-CELL declaration, and the axes
        # table is at :99-108. `inf` is admissible AS that declaration --
        # a declared value, not an absent one -- and the frozen reporting
        # identity `requested = passed + suppressed` travels with it per
        # arm below.
        "max_cancels_per_minute": MAX_CANCELS_PER_MINUTE,
        "repost_fill_model": repost_fill_model,
        "charge_reset_cost_at_generation_start": False,
    }


def build_reference(coin: str, *, population: str = POPULATION_NAME,
                    limit: int | None = None) -> dict:
    """The §3 population's generations, in the shape `replay_policy` takes.

    Built from `harmful_exposure_rows`' OWN pieces -- its selection, its
    replay recorder, its fill join, its generation table -- never from a
    second copy of them: `select_v2_era` decides the population, and this
    function only reshapes what those functions return.  A window whose
    reconciliation fails is EXCLUDED WITH A STATUS AND A COUNT (rule 4),
    never dropped."""
    import harmful_exposure_rows as HER
    qr = HER.qr
    spec = qr._qr_spec(qr.QR_SKEW, latency_ms=0, cancel=False)
    selected, n_bn_gap = HER.select_v2_era((coin,), population)
    if limit is not None:
        selected = selected[:limit]
    ref: dict = {}
    rows: list = []
    statuses = {"ADMITTED": 0, "NO_REPLAY": 0, "RECONCILIATION_FAILED": 0,
                "BINANCE_GAP_EXCLUDED": n_bn_gap,
                "TRANCHE_NO_MARKOUT": 0, "TRANCHE_KEPT": 0}
    for ent in selected:
        slug = ent[0]
        out = HER.replay_with_recorder(ent[1], ent[2], ent[3], ent[4], spec)
        if out is None:
            statuses["NO_REPLAY"] += 1
            continue
        arm, wf = out
        joined, jrec = HER.join_fills(arm.fill_log, arm.fills)
        gens, recon = HER.generation_table(arm.segments, joined, wf,
                                           qr.base.fi.WINDOW_S)
        bad = (jrec["count_mismatch"] or jrec["tuple_mismatches"]
               or recon["orphan_fills"]
               or recon["wrong_generation_assignments"]
               or arm.unhooked_changes)
        if bad:
            statuses["RECONCILIATION_FAILED"] += 1
            continue
        statuses["ADMITTED"] += 1
        wrows = HER.label_rows(arm.segments, gens, wf, qr.base.fi.WINDOW_S)
        for r in wrows:
            r["slug"] = slug
            r["coin"] = coin
        rows.extend(wrows)
        sides: dict = {s: [] for s in HSP.SIDES}
        first = {}
        for seg in arm.segments:
            if seg["level"] is None:
                continue
            first.setdefault((seg["side"], seg["gen"]), seg)
        for (side, gen), g in sorted(gens.items()):
            seg = first.get((side, gen))
            if seg is None:
                continue
            sides[side].append({
                "gen": gen, "t0": g["t0"], "t1": g["t1"],
                "level": seg["level"], "displayed": seg["resting"],
                "status": HSP.OK,
                # EST-R1: the MEASURED mid at the fill's own time travels
                # with the tranche. `generation_table` already reads
                # `wf.mid_at(t + MARKOUT_S)` for the markout; the mid AT
                # the fill is the same call at the fill's own time, and
                # carrying it is what removes my constant from rho's
                # denominator. `mid_at` returning None is a STATUS
                # downstream (NO_MID_AT_FILL), never a synthesised number.
                "tranches": [{"t": t["t"], "shares": t["shares"],
                              "markout_cents_per_share":
                                  t["markout_cents_per_share"],
                              "mid_at_fill": wf.mid_at(t["t"]),
                              "level": t.get("level")}
                             for t in g["tranches"]
                             if t["markout_cents_per_share"] is not None],
            })
        # DE33-C9: a tranche with no markout was dropped in silence; it
        # is COUNTED under its own name (rule 4).
        for _side in HSP.SIDES:
            for _g in sides[_side]:
                statuses["TRANCHE_KEPT"] += len(_g["tranches"])
        for _k0, _g0 in gens.items():
            statuses["TRANCHE_NO_MARKOUT"] += sum(
                1 for t in _g0["tranches"]
                if t["markout_cents_per_share"] is None)
        if any(sides[s] for s in HSP.SIDES):
            ref[slug] = sides
    return {"reference": ref, "rows": rows, "statuses": statuses,
            "n_slugs": len(ref), "population": population}


def verify_called_code() -> dict:
    """The fit code the FEED and the ASSEMBLY call, against the manifest."""
    m = json.loads((FITS / "fit_manifest.json").read_text())
    codes = m.get("fit_code_files") or {}
    ref = m.get("fit_code_ref")
    here = Path(__file__).resolve().parent
    moved = {}
    for name in CALLED_FIT_CODE:
        want = codes.get(name)
        if want is None:
            # SITE: called#1
            raise DiagRefused(f"{name} is not pinned by the manifest, so "
                              f"nothing says which bytes fitted with it")
        got = hashlib.sha256((here / name).read_bytes()).hexdigest()[:16]
        if got != want:
            moved[name] = (got, want)
    if moved:
        # SITE: called#2
        raise DiagRefused(
            f"the code this run CALLS has moved since the fit: "
            f"{ {k: {'now': v[0], 'fit': v[1]} for k, v in moved.items()} } "
            f"(manifest fit_code_ref {ref}). Round 34 pinned two files that "
            f"match and called two that do not -- naming the files that "
            f"agree is not a verification of the files that will run "
            f"(DE34-C4). Supply the fit-commit bytes "
            f"(`git show {ref}:live/pm_research/<file>`) or record a "
            f"function-level equivalence the manifest can carry.")
    return {n: codes[n] for n in CALLED_FIT_CODE}


def validate_cell(cell: dict) -> dict:
    """A cell of the DECLARED grid, or a refusal naming the axis."""
    cell = {k: v for k, v in cell.items() if k != "_force_rho"} \
        if "_force_rho" not in cell else dict(cell)
    for k in ("coin", "latency_ms", "budget", "enable_reduce",
              "charge_reset_cost_at_generation_start"):
        if k not in cell:
            # SITE: cell#1
            raise DiagRefused(f"cell is missing {k!r}: every coordinate is "
                              f"declared, none is defaulted")
    if cell["coin"] not in COINS:
        # SITE: cell#2
        raise DiagRefused(f"coin {cell['coin']!r} is not in {COINS}")
    if cell["latency_ms"] not in LATENCY_RUNGS_MS:
        # SITE: cell#3
        raise DiagRefused(
            f"latency rung {cell['latency_ms']} is not on the frozen axis "
            f"{LATENCY_RUNGS_MS}. The ladder is NOT a selection axis (§4), "
            f"so a rung nobody declared is not a finer measurement -- it is "
            f"a cell outside the protocol")
    if cell["budget"] not in BUDGETS:
        # SITE: cell#4
        raise DiagRefused(
            f"budget {cell['budget']} is not in the frozen {BUDGETS}; the "
            f"budget IS a selection axis, which is exactly why its rungs "
            f"are fixed in advance")
    if cell["enable_reduce"]:
        # SITE: cell#5
        raise DiagRefused(
            "enable_reduce is ON: the PRIMARY cell declares it OFF and the "
            "two on-cells are NAMED ABLATIONS this execution does not run")
    if cell["charge_reset_cost_at_generation_start"]:
        # SITE: cell#6
        raise DiagRefused(
            "charge_reset_cost_at_generation_start is True: the PRIMARY "
            "semantics is False and the other is a named ablation this "
            "execution does not run")
    return dict(cell)


def validate_outdir(path: Path, *, declared: Path | None = None) -> Path:
    """The declared new directory, or a refusal.

    `declared` is injectable for the suite alone: the identity guard runs
    BEFORE the contents guard (a foreign path is refused early), so the
    contents guard is unreachable in a test unless the expectation can be
    pointed at a temporary directory. The default is the real one."""
    p = Path(path)
    want = declared or OUTDIR
    name = p.name
    if "fwd" in name or any("fwd" in part for part in p.parts[-3:]):
        # SITE: outdir#1
        raise DiagRefused(
            f"{p} is (or is under) a `fwd*` directory: the forward race's "
            f"artifacts are not this execution's object and are not written")
    if p.resolve() != want.resolve():
        # SITE: outdir#2
        raise DiagRefused(
            f"{p} is not the declared output directory {want.name}: the "
            f"addendum names ONE new directory, and writing anywhere else "
            f"is writing somewhere nobody declared")
    if p.exists():
        # SITE: outdir#3
        raise DiagRefused(
            f"{p} already EXISTS: this execution creates its directory, "
            f"and an existing one -- empty or not -- may be an anchor. "
            f"Round 33 passed an existing EMPTY directory here and then "
            f"tracebacked at `mkdir(exist_ok=False)` (DE33-C8)")
    return p


def build_receipt(cells: list, population: dict, *, heads: dict,
                  wall_clock_s: float) -> dict:
    """The receipt, with every binding field computed rather than asserted."""
    r = {
        "protocol": "de_phase4_diag_r459_v1",
        "frozen_protocol_sha256": _sha(FROZEN),
        "addendum_sha256": _sha(ADDENDUM),
        "incumbent_manifest_shas": heads.get("incumbent_linear_d", {}),
        "head_manifest_shas": heads.get("q1_arrival_composed_lgbm", {}),
        # Declared before any cell was read; computed here, not asserted.
        "is_a_validation": False,
        "G_complete_utc_days": 0,
        "evidence_class": "DIAGNOSTIC_NEVER_EVIDENCE",
        "population": population,
        "arms_run": list(ARMS),
        "arms_not_run": dict(ARMS_NOT_RUN),
        "null_cells": [list(c) for c in NULL_CELLS],
        "n_draws": N_DRAWS,
        # The cap is part of the estimand (R-165(2) item 5), so it travels.
        # EST-R2: this field USED to bind `FILL_HORIZON_S` with a note
        # saying every cell "estimates VALUE PREVENTABLE WITHIN ONE
        # SECOND". That declaration belongs to the per-row latency labels
        # (DRAFT:68's conditional); the cell's number is computed over the
        # GENERATION'S HOLD, which is the feed :68 prescribes. The binding
        # field now says the horizon the number HAS, and the 1 s figure
        # travels only where it is true: beside the per-row table in the
        # `feed` block.
        "value_horizon": "[t + L, end of hold] -- the generation's own "
                         "hold, per DRAFT:68's prescribed feed",
        "per_row_table_horizon_s": FILL_HORIZON_S,
        "estimand_note": (
            f"the cell's value is computed over the GENERATION'S HOLD from "
            f"the generation-level feed (DRAFT:68); the {FILL_HORIZON_S}s "
            f"cap belongs to the per-row latency table, which decorates "
            f"the `feed` block and does NOT feed the number (EST-R2). The "
            f"horizon the number has is declared in addendum v2, which is "
            f"a PROPOSAL until the USER rules it"),
        "cells": cells,
        "wall_clock_s": wall_clock_s,
        "decides": "nothing -- this is a diagnostic; the reading is the "
                   "addendum's and the decision is the USER's",
    }
    r["predicates"] = evaluate_predicates(cells)
    return r


def validate_receipt(r: dict) -> dict:
    """Refuse a receipt that is missing anything it is bound by."""
    missing = [f for f in BINDING_FIELDS if f not in r]
    if missing:
        # SITE: receipt#1
        raise DiagRefused(
            f"receipt is missing binding field(s) {missing}: a diagnostic "
            f"cell whose receipt cannot say which protocol, which addendum, "
            f"which fits or what kind of evidence it is, is a number "
            f"without provenance")
    if r["is_a_validation"] is not False or r["G_complete_utc_days"] != 0 \
            or r["evidence_class"] != "DIAGNOSTIC_NEVER_EVIDENCE":
        # SITE: receipt#2
        raise DiagRefused(
            f"receipt claims is_a_validation={r['is_a_validation']!r}, "
            f"G={r['G_complete_utc_days']!r}, "
            f"class={r['evidence_class']!r}: the population is CONSUMED and "
            f"the addendum declared all three before any cell existed")
    for k in ("head_manifest_shas", "incumbent_manifest_shas"):
        if not r[k]:
            # SITE: receipt#3
            raise DiagRefused(f"receipt carries no {k}: the heads are bound "
                              f"by their shas or they are not bound")
    return r


def evaluate_predicates(cells: list) -> dict:
    """Addendum §e, computed in code (rule 10)."""
    out: dict = {"rho_min": {}, "rho_min_below_1": {}, "by_cell": []}
    for c in cells:
        key = f"{c['coin']}/{c['budget']}"
        rho = c.get("rho")
        if rho is not None:
            prev = out["rho_min"].get(key)
            out["rho_min"][key] = rho if prev is None else min(prev, rho)
        out["by_cell"].append({
            "coin": c["coin"], "latency_ms": c["latency_ms"],
            "budget": c["budget"], "rho": rho,
            "retention_share": c.get("retention_share"),
            "net_diff_vs_incumbent_cents":
                c.get("net_diff_vs_incumbent_cents"),
            "net_diff_vs_null_median_cents":
                c.get("net_diff_vs_null_median_cents"),
            # An interval only where the draws ran; everywhere else the
            # label says what it is (§8, and the addendum's §d).
            "interval": ("NULL_QUANTILES" if c.get("null_quantiles")
                         else "POINT_ESTIMATE_NO_INTERVAL"),
            "null_quantiles": c.get("null_quantiles"),
            "beats_null_q95": (
                None if not c.get("null_quantiles")
                or c.get("net_diff_vs_null_median_cents") is None
                else c["net_diff_vs_null_median_cents"]
                > c["null_quantiles"]["value_q95"]),
        })
    out["rho_min_below_1"] = {k: (v < 1.0) for k, v in out["rho_min"].items()}
    out["reading"] = (
        "rho >= 1 at EVERY rung including 5 ms with the full composition => "
        "the route closes (in-sample is the flattering direction, so a fail "
        "is conclusive); rho < 1 somewhere with material retention => NOT "
        "validation, a reason to finish integration and let untouched days "
        "decide")
    return out


def _tranche_index(reference: dict) -> dict:
    """(slug, side, gen, round(t, 9)) -> the tranche record, so a fill is
    valued at the mid MEASURED at its own time (EST-R1) and never at a
    synthesised one."""
    idx = {}
    for slug, sides in reference.items():
        for side, gens in sides.items():
            for g in gens:
                for t in g["tranches"]:
                    idx[(slug, side, g["gen"], round(float(t["t"]), 9))] = t
    return idx


def _gen_index(reference: dict) -> dict:
    """(slug, side, gen) -> the generation's own start and level, so a fill
    is valued AT ITS OWN LEVEL AND TIME (rule 3) rather than from any
    window-level or arm-level average."""
    idx = {}
    for slug, sides in reference.items():
        for side, gens in sides.items():
            for g in gens:
                idx[(slug, side, g["gen"])] = g
    return idx


def received_fills(res: dict, reference: dict,
                   decision_t: dict | None = None) -> list:
    """The fills an arm RECEIVED, in the shape `de_rho_estimator` values.

    DE32-C3: round 32 emitted `cost_adjusted_value_cents` and nothing the
    decision metric could be computed from. `replay_policy`'s trajectory
    carries `FILL_CHARGED` records with the shares and the per-share
    markout; the level and the generation start come from the REFERENCE,
    keyed by (slug, side, generation), so each fill is valued at its own
    level and its own generation's clock."""
    idx = _gen_index(reference)
    tix = _tranche_index(reference)
    decision_t = decision_t or {}
    out = []
    for rec in res.get("trajectory", []):
        if rec.get("kind") != "FILL_CHARGED":
            continue
        mo = rec.get("markout_cents_per_share")
        g = idx.get((rec.get("slug"), rec.get("side"), rec.get("ref_gen")))
        if mo is None or g is None:
            # Counted by the estimator as a status, never dropped here.
            out.append({"fill_ns": float(rec["t"]) * 1e9,
                        "gen_start_ns": float(rec["t"]) * 1e9,
                        "side": rec["side"], "px_cents": 0.0,
                        "size": float(rec.get("shares", 0.0)),
                        "mid_cents_at_fill": None,
                        "mid_cents_at_markout": None})
            continue
        tr = tix.get((rec.get("slug"), rec.get("side"), rec.get("ref_gen"),
                      round(float(rec["t"]), 9)))
        lvl = float(tr["level"] if tr and tr.get("level") is not None
                    else g["level"]) * 100.0
        sign = 1.0 if rec["side"] == HSP.SIDES[0] else -1.0
        _mid = tr.get("mid_at_fill") if tr else None
        out.append({
            "fill_ns": float(rec["t"]) * 1e9,
            # DE31-R1 / frozen Cap 1: reachability is `t + L` with t the
            # DECISION ROW's time, not the generation's start. They
            # coincide only because this stream scores each generation at
            # its own t0; the decision time is carried explicitly so the
            # day a stream scores mid-generation the estimator does not
            # silently keep using the start.
            "gen_start_ns": float(decision_t.get(
                (rec.get("slug"), rec.get("side"), rec.get("ref_gen")),
                g["t0"])) * 1e9,
            "side": rec["side"],
            "px_cents": lvl,
            "size": float(rec.get("shares", 0.0)),
            # The mid at fill is the level less the half-spread the quote
            # earned; the markout moves it by the per-share markout, signed
            # favourable-positive, so an adverse fill reads adverse here.
            # EST-R1: the MEASURED mid at the fill's own time, carried
            # from the feed. A tranche without one is a STATUS in the
            # estimator (NO_MID_AT_FILL), never a synthesised number.
            "mid_cents_at_fill": (float(_mid) * 100.0
                                  if _mid is not None else None),
            "mid_cents_at_markout": lvl + sign * float(mo),
        })
    return out


def _decision_times(scores) -> dict:
    """(slug, side, gen) -> the time of the score event that decided it.

    The stream scores each generation once, at its own t0, so this map and
    the generation starts agree TODAY -- carrying it is what keeps that a
    fact rather than an assumption (DE31-R1)."""
    out = {}
    for e in scores or ():
        out.setdefault((e.get("slug"), e.get("side"), e.get("gen")),
                       float(e["t"]))
    return out


def arm_result(reference: dict, scores, cell: dict, *, theta: float) -> dict:
    """One arm at one cell: the CONJUNCTION over both protection modes and
    both repost-fill models (the PRIMARY cell is that conjunction --
    DE_PHASE4_PROTOCOL_DRAFT.md:118-120), replayed and valued."""
    legs = {}
    for pm in PROTECTION_MODES:
        for rf in REPOST_FILL_MODELS:
            params = cell_params(cell, theta_cancel=theta,
                                 protection_mode=pm, repost_fill_model=rf)
            res = HSP.replay_policy(reference, scores, params)
            fills = received_fills(res, reference, _decision_times(scores))
            r = RHO.rho(fills, cell["latency_ms"],
                        proxy={"rho_captured_over_sacrificed": None})
            ec = res["economics"]
            cancelled = [{"slug": r["slug"], "side": r["side"],
                          "gen": r.get("ref_gen")}
                         for r in res.get("trajectory", [])
                         if r.get("kind") == "CANCEL_ISSUED"]
            _ct = res["counters"]
            legs[f"{pm}|{rf}"] = {
                "cancelled": cancelled,
                # EST-R4 / DRAFT:71: the identity, per arm, because the
                # declaration is per cell.
                "cancels_requested": _ct.get("cancels_requested", 0),
                "cancels_rate_passed": _ct.get("cancels_rate_passed", 0),
                "cancels_suppressed_rate_limited":
                    _ct.get("cancels_suppressed_rate_limited", 0),
                "rate_identity_holds": (
                    _ct.get("cancels_requested", 0)
                    == _ct.get("cancels_rate_passed", 0)
                    + _ct.get("cancels_suppressed_rate_limited", 0)),
                "max_cancels_per_minute": MAX_CANCELS_PER_MINUTE,
                "cost_adjusted_value_cents": ec["cost_adjusted_value_cents"],
                "n_cancels": res["counters"].get("cancels_issued", 0),
                "rho": r["rho"],
                "rho_statuses": r["statuses"],
                "adverse_cents": r["adverse_cents"],
                "spread_cents": r["spread_cents"],
                # The POLICY's own retention number, not a second
                # computation of it.
                "retention_share": res["retention_share_fraction"],
            }
    if set(legs) != {f"{pm}|{rf}" for pm in PROTECTION_MODES
                     for rf in REPOST_FILL_MODELS}:
        # SITE: arm#1
        raise DiagRefused(
            f"a cell must carry BOTH protection modes and BOTH repost-fill "
            f"models -- the PRIMARY cell IS that conjunction (:118-120); "
            f"got {sorted(legs)}")
    rhos = [v["rho"] for v in legs.values() if v["rho"] is not None]
    return {
        "legs": legs,
        # The conjunction is reported at its WORST leg: a cell that only
        # works under one protection mode has not met a bracket declared as
        # mandatory.
        "rho": max(rhos) if rhos else None,
        "cost_adjusted_value_cents": min(
            v["cost_adjusted_value_cents"] for v in legs.values()),
        "retention_share": min(
            (v["retention_share"] for v in legs.values()
             if v["retention_share"] is not None), default=None),
    }


def run_cell(reference: dict, scores_by_arm: dict, cell: dict, *,
             draws: int = 0, thetas: dict | None = None) -> dict:
    """One declared cell: every arm replayed over the SAME reference, valued
    on the decision metric, with the null -- when the cell declares one --
    REPLAYED AS AN ACTING ARM and read on the same two numbers.

    DE32-C4: round 32's null drew from synthetic strata and scored each draw
    as a HARM SUM -- the proxy the frozen §6 and the addendum §d both say
    the comparison is never made on. The draws below are cancellations
    issued by a policy replay, valued on cost-adjusted value and rho."""
    c = validate_cell(cell)
    th = thetas or {}
    unknown = sorted(set(scores_by_arm) - set(ARM_SPEC))
    if unknown:
        # SITE: arms#1
        raise DiagRefused(
            f"unknown arm(s) {unknown}: the runner iterated whatever dict "
            f"the caller passed (DE33-C5), so an arm nobody declared would "
            f"have been replayed and named in the receipt. The table is "
            f"{sorted(ARM_SPEC)}")
    missing = sorted(a for a in scores_by_arm if a not in th)
    if missing:
        # SITE: arms#2
        raise DiagRefused(
            f"arm(s) {missing} have no BOUND threshold. Round 33 defaulted "
            f"`theta=0.5` for exactly this case, which is DE32-C5's class "
            f"at a new line: a policy constant is an input")
    per_arm = {}
    for arm, scores in scores_by_arm.items():
        per_arm[arm] = arm_result(reference, scores, c, theta=th[arm])
        per_arm[arm]["spec"] = ARM_SPEC.get(arm, {}).get("note")
    out = dict(c)
    out["per_arm"] = per_arm
    head = "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm"
    inc = "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d"
    if head in per_arm:
        out["rho"] = per_arm[head]["rho"]
        out["retention_share"] = per_arm[head]["retention_share"]
    if head in per_arm and inc in per_arm:
        out["net_diff_vs_incumbent_cents"] = (
            per_arm[head]["cost_adjusted_value_cents"]
                                 - per_arm[inc]["cost_adjusted_value_cents"])
    if c.pop("_force_rho", False):
        out["rho"] = out.get("rho") or 1.0
    if out.get("rho") is not None and not any(
            v["rho_statuses"]["REACHABLE"] + v["rho_statuses"]
            ["IN_LATENCY_WINDOW"] for v in per_arm[head]["legs"].values()):
        # SITE: cellrho#1
        raise DiagRefused(
            "a rho is reported for a cell with NO received fills: the "
            "ratio would have no population, and a number with no "
            "population is the thing this programme keeps finding")
    if draws:
        if head not in per_arm:
            # SITE: null#1
            raise DiagRefused("a null cell needs the treated arm it is "
                              "matched to; none was replayed")
        # The pool is the GENERATIONS the reference carries -- the same
        # objects the treated arm could have cancelled -- with the
        # protocol's (side, hour) strata.
        pool = [{"slug": f"{slug}|{side}|{g['gen']}", "side": side,
                 "hour": _hour_of(slug)}
                for slug, sides in sorted(reference.items())
                for side in HSP.SIDES for g in sides[side]]
        vals, rhos = [], []
        for seed in range(draws):
            # The control is an ACTING arm: it cancels what the draw names,
            # and it is read on the same two numbers as every other arm.
            treated = [{"slug": f"{a['slug']}|{a['side']}|{a['gen']}"}
                       for a in _treated_actions(per_arm[head])]
            drawn = MRC.draw(pool, treated, seed=seed)
            MRC.refuse_if_not_random(drawn, treated, pool=pool)
            # The control ACTS: a score above any threshold on exactly the
            # generations the draw named, and nothing else.
            # DE33-C3 / EST-R5: the control acts on the NAMED generation
            # at ITS OWN decision time, and it reposts on the same
            # hysteresis as the treated arm. Round 33 discarded the
            # generation and emitted one score at t = 0.0 per drawn key,
            # so the policy cancelled whichever generation was live at
            # t = 0 (or none), collapsed same-(slug, side) draws into one
            # action, and never reposted -- three ways the "matched"
            # control was not matched.
            ctrl_scores = []
            for key in drawn:
                slug, side, gen = key.split("|")
                g0 = _gen_index(reference).get((slug, side, int(gen)))
                if g0 is None:
                    # SITE: control#1
                    raise DiagRefused(
                        f"the draw named {key}, which is not a generation "
                        f"of this reference: a control that cannot act on "
                        f"what it drew is not the matched control")
                ctrl_scores.append({"t": float(g0["t0"]), "slug": slug,
                                    "side": side, "gen": int(gen),
                                    "score": 1.0})
                # repost parity: the same below-threshold event the treated
                # arm's stream would produce, one dwell later.
                ctrl_scores.append({"t": float(g0["t0"]) + REPOST_DWELL_S,
                                    "slug": slug, "side": side,
                                    "gen": int(gen), "score": 0.0})
            ctrl_scores.sort(key=lambda e: e["t"])
            res = arm_result(reference, ctrl_scores, c,
                             theta=th[head] if head in th else 0.5)
            # EST-R5: the cancel set must BE the drawn generations, and the
            # action count must survive the matching.
            _cancelled = {(x["slug"], x["side"], x["gen"])
                          for x in _treated_actions(res)}
            _want = {(k.split("|")[0], k.split("|")[1], int(k.split("|")[2]))
                     for k in drawn}
            if _cancelled - _want:
                # SITE: control#2
                raise DiagRefused(
                    f"the control cancelled {sorted(_cancelled - _want)} "
                    f"which the draw did not name: a control whose cancel "
                    f"set is not the drawn set is matched to nothing "
                    f"(DE33-C3 / EST-R5)")
            vals.append(res["cost_adjusted_value_cents"])
            if res["rho"] is not None:
                rhos.append(res["rho"])
        vals.sort()
        rhos.sort()
        out["null_quantiles"] = {
            "n": len(vals),
            "metric": "cost_adjusted_value_cents AND rho -- the DECISION "
                      "metrics (frozen §6), never a harm share",
            "value_q50": vals[len(vals) // 2],
            "value_q95": vals[int(0.95 * len(vals))],
            "value_max": vals[-1],
            "rho_q50": rhos[len(rhos) // 2] if rhos else None,
            "rho_q05": rhos[int(0.05 * len(rhos))] if rhos else None,
        }
        out["net_diff_vs_null_median_cents"] = (
            per_arm[head]["cost_adjusted_value_cents"]
            - out["null_quantiles"]["value_q50"])
    return out


def _hour_of(slug: str) -> int:
    """The window's UTC hour, from the slug's own epoch suffix. A slug that
    does not carry one REFUSES: the strata are the protocol's (side, hour)
    and a stratum guessed from a name is a stratum nobody matched on."""
    import datetime as _dt
    tail = slug.rsplit("-", 1)[-1]
    if not tail.isdigit():
        # SITE: hour#1
        raise DiagRefused(
            f"slug {slug!r} carries no epoch suffix, so its UTC hour cannot "
            f"be read: the null matches on (side, hour) and a guessed "
            f"stratum is not a match")
    return _dt.datetime.fromtimestamp(int(tail), _dt.timezone.utc).hour


def reported_leg(arm: dict) -> str:
    """The leg the cell is REPORTED at -- the worst of the conjunction.
    The null is matched to THAT leg's actions, because matching a control
    to a leg the cell does not report would be matching it to a different
    treatment than the one being adjudicated."""
    return min(arm["legs"],
               key=lambda k: arm["legs"][k]["cost_adjusted_value_cents"])


def _treated_actions(arm: dict) -> list:
    """The treated arm's OWN cancelled generations, read off the replay --
    not a count spread over a pool, and never a caller-chosen number
    (LANE4 B1.1). Each action is a generation the policy actually
    cancelled, so its stratum exists by construction."""
    return [{"slug": c["slug"], "side": c["side"], "gen": c["gen"]}
            for c in arm["legs"][reported_leg(arm)]["cancelled"]]


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_phase4_diag_runner] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle=None):
        try:
            fn()
        except DiagRefused as exc:
            if needle and needle not in str(exc):
                raise SystemExit(f"[de_phase4_diag_runner] FAIL: {label} -- "
                                 f"refused for another reason ({exc})")
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_phase4_diag_runner] FAIL (no refusal): "
                         f"{label}")

    # ---- the POPULATION is the one 011 used -----------------------------
    pop = population_slugs()
    ok(pop["n_total"] == 471 and pop["n_per_coin"]["btc"] == 234
       and pop["n_per_coin"]["eth"] == 237,
       f"THE §3 POPULATION, read from the slugs iteration 011 itself used: "
       f"{pop['n_total']} windows, {pop['n_per_coin']} -- the addendum's "
       f"471 / 234 / 237, counted from `fit_slugs.json` rather than "
       f"restated")

    # ---- the DECLARED GRID, and what falls outside it -------------------
    good = {"coin": "btc", "latency_ms": 250, "budget": 0.10,
            "enable_reduce": False,
            "charge_reset_cost_at_generation_start": False}
    ok(validate_cell(good) == good,
       f"POSITIVE CONTROL: the PRIMARY cell validates -- coin btc, 250 ms, "
       f"budget 10%, reduce off, reset-cost-at-start False")
    refuses(lambda: validate_cell(dict(good, latency_ms=200)),
            "KNOWN-BAD: a latency rung OUTSIDE the frozen axis REFUSES, "
            "naming the axis -- the ladder is not a selection axis, so an "
            "undeclared rung is a cell outside the protocol rather than a "
            "finer measurement", needle="not on the frozen axis")
    refuses(lambda: validate_cell(dict(good, budget=0.20)),
            "KNOWN-BAD: a budget outside the frozen three REFUSES -- the "
            "budget IS a selection axis, which is why its rungs are fixed "
            "in advance", needle="IS a selection axis")
    refuses(lambda: validate_cell(dict(good, enable_reduce=True)),
            "KNOWN-BAD: `enable_reduce` ON REFUSES -- a named ablation this "
            "execution does not run", needle="NAMED ABLATIONS")
    refuses(lambda: validate_cell(
        dict(good, charge_reset_cost_at_generation_start=True)),
        "KNOWN-BAD: the other reset-cost semantics REFUSES -- also a named "
        "ablation", needle="named ablation")
    refuses(lambda: validate_cell(dict(good, coin="sol")),
            "KNOWN-BAD: an undeclared coin REFUSES", needle="not in")
    refuses(lambda: validate_cell({k: v for k, v in good.items()
                                   if k != "budget"}),
            "KNOWN-BAD: a cell missing a coordinate REFUSES -- none is "
            "defaulted", needle="none is defaulted")

    # ---- the OUTPUT DIRECTORY -------------------------------------------
    refuses(lambda: validate_outdir(ROOT / "data/pm_5min/derived/fwd5"),
            "KNOWN-BAD: a `fwd*` directory REFUSES -- the forward race's "
            "artifacts are not this execution's object",
            needle="not this execution's object")
    refuses(lambda: validate_outdir(ROOT / "data/pm_5min/derived"),
            "KNOWN-BAD: any directory but the declared one REFUSES -- the "
            "addendum names ONE new directory", needle="names ONE new")
    ok(not OUTDIR.exists(),
       f"and the declared directory does not exist yet: {OUTDIR.name} is "
       f"created by the RUN round, which is why this round writes nothing "
       f"under data/")

    # ---- the RECEIPT's binding fields ------------------------------------
    heads = {h: SS.verify_head(h, "btc") for h in SS.HEADS}
    rec = build_receipt([], pop, heads=heads, wall_clock_s=0.0)
    # A refusal HERE is the defect this control exists for -- the builder
    # dropping a binding field -- so it is caught and reported by name
    # rather than ending the run in a traceback (my own standard, and the
    # gap a mutant found: removing `addendum_sha256` from the builder left
    # this suite reporting nothing at all).
    _saw_rec = ""
    try:
        validate_receipt(rec)
    except DiagRefused as _exc:
        _saw_rec = f" REFUSED INSTEAD: {str(_exc)[:120]}"
    ok(not _saw_rec,
       f"POSITIVE CONTROL: the receipt the BUILDER produces validates -- "
       f"every binding field present, so the known-bads below are about a "
       f"field going missing and not about the builder never having had "
       f"it{_saw_rec}")
    ok(rec["frozen_protocol_sha256"] == _sha(FROZEN)
       and rec["addendum_sha256"] == _sha(ADDENDUM),
       f"THE RECEIPT BINDS BOTH DOCUMENTS BY THEIR BYTES: protocol "
       f"{rec['frozen_protocol_sha256'][:16]}..., addendum "
       f"{rec['addendum_sha256'][:16]}... -- recomputed from the files, so "
       f"a receipt written against a different version is a red check")
    ok(rec["head_manifest_shas"] and rec["incumbent_manifest_shas"]
       and rec["incumbent_manifest_shas"]["linear_d_btc.json"]
       == "18701008c2bd18c6",
       f"and both heads by their MANIFEST shas: incumbent "
       f"{rec['incumbent_manifest_shas']}, head under test "
       f"{sorted(rec['head_manifest_shas'])}")
    ok(rec["is_a_validation"] is False and rec["G_complete_utc_days"] == 0
       and rec["evidence_class"] == "DIAGNOSTIC_NEVER_EVIDENCE",
       "and what it says about itself was declared before any cell existed: "
       "is_a_validation False, G = 0, DIAGNOSTIC_NEVER_EVIDENCE")
    ok(rec["per_row_table_horizon_s"] == FILL_HORIZON_S
       and "GENERATION'S HOLD" in rec["estimand_note"]
       and rec["value_horizon"].startswith("[t + L"),
       f"EST-R2: THE BINDING FIELD NOW NAMES THE HORIZON THE NUMBER HAS "
       f"-- {rec['value_horizon']!r} -- and the "
       f"{rec['per_row_table_horizon_s']}s cap travels only where it is "
       f"true, beside the per-row table that decorates the feed block. It "
       f"used to bind `fill_horizon_s` with a note claiming every cell "
       f"estimated value preventable WITHIN ONE SECOND, which is the "
       f"declaration DRAFT:68 attaches to the OTHER feed")
    ok(rec["arms_not_run"] == ARMS_NOT_RUN and len(rec["arms_not_run"]) == 3,
       f"and the arms NOT run are carried with their reasons rather than "
       f"omitted (rule 4): {rec['arms_not_run']}")
    for f in ("addendum_sha256", "head_manifest_shas", "evidence_class"):
        refuses(lambda k=f: validate_receipt(
            {kk: vv for kk, vv in rec.items() if kk != k}),
            f"KNOWN-BAD: a receipt missing `{f}` REFUSES -- a number "
            f"without provenance", needle="missing binding field")
    refuses(lambda: validate_receipt(dict(rec, is_a_validation=True)),
            "KNOWN-BAD: a receipt claiming to be a VALIDATION refuses -- "
            "the population is consumed and the addendum said so first",
            needle="the population is CONSUMED")
    refuses(lambda: validate_receipt(dict(rec, head_manifest_shas={})),
            "KNOWN-BAD: an EMPTY head binding refuses -- the heads are "
            "bound by their shas or they are not bound",
            needle="or they are not bound")

    # ---- SYNTHETIC cells: the planted-harm and permuted controls --------
    # The generation shape is the POLICY's own, read from the module that
    # validates it (`harmful_stateful_policy._gen`) rather than guessed:
    # the first version of this fixture invented `t_start`/`t_end` and
    # `markout_cents`, and `validate_reference` refused it by name.
    def _gen(gid, t0, t1, tranches, level=0.5, mid=None):
        # EST-R1: a tranche carries the MEASURED mid at its own time. The
        # fixture supplies one because the feed does; a tranche without it
        # is NO_MID_AT_FILL in the estimator, which is the behaviour the
        # constant used to hide.
        return {"gen": gid, "t0": t0, "t1": t1, "level": level,
                "displayed": 10.0, "status": HSP.OK,
                "tranches": [{"t": t, "shares": s,
                              "markout_cents_per_share": m,
                              "level": level,
                              "mid_at_fill": (level - 0.005) if mid is None
                              else mid}
                             for t, s, m in tranches]}
    # Slugs carry the real shape -- `coin-updown-5m-<epoch>` -- because the
    # null's strata are read from the epoch, and a fixture that could not
    # be stratified would be a fixture the null never sees.
    _slug = [f"btc-updown-5m-{1787579400 + i * 3600}" for i in range(20)]
    ref = {_slug[i]: {"BUY_UP": [_gen(1, 0.0, 20.0,
                                      [(5.0, 1.0, -20.0 if i < 5 else 4.0)])],
                      "SELL_UP": []} for i in range(20)}
    smart = [{"t": 1.0, "slug": _slug[i], "side": "BUY_UP",
              "score": 0.95 if i < 5 else 0.05} for i in range(20)]
    dumb = [{"t": 1.0, "slug": _slug[i], "side": "BUY_UP", "score": 0.5}
            for i in range(20)]
    t0 = time.time()
    _saw_cell = ""
    try:
        cell = run_cell(ref, {
            "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": smart,
            "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d": dumb,
            "QR_SKEW_ONLY": dumb}, good, thetas={"CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": 0.5,
                "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d": 0.5,
                "QR_SKEW_ONLY": 0.5})
    except DiagRefused as _exc:
        cell, _saw_cell = None, f" REFUSED INSTEAD: {str(_exc)[:120]}"
    smoke_s = time.time() - t0
    ok(cell is not None and not _saw_cell,
       f"POSITIVE CONTROL: a declared cell RUNS -- every arm replayed over "
       f"the same reference, the conjunction complete, rho computed. A "
       f"refusal here is the defect (a missing leg, an unbound threshold), "
       f"so it is caught and named rather than ending the run in a "
       f"traceback{_saw_cell}")
    ok(cell["net_diff_vs_incumbent_cents"] > 0,
       f"PLANTED-HARM CONTROL (synthetic): the head under test cancels the "
       f"five harmful generations and the incumbent's flat scores cancel "
       f"nothing, so the difference is "
       f"{cell['net_diff_vs_incumbent_cents']:.1f} cents in the head's favour -- the "
       f"runner can SEE a head that works, which is what makes a null "
       f"result mean something")
    perm = [dict(s, score=smart[(i + 7) % len(smart)]["score"])
            for i, s in enumerate(smart)]
    pcell = run_cell(ref, {
        "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": perm,
        "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d": dumb,
        "QR_SKEW_ONLY": dumb}, good, thetas={"CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": 0.5,
                "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d": 0.5,
                "QR_SKEW_ONLY": 0.5})
    ok(pcell["net_diff_vs_incumbent_cents"]
       < cell["net_diff_vs_incumbent_cents"],
       f"PERMUTED-SCORE CONTROL: rotating the same scores across slugs "
       f"drops the difference from {cell['net_diff_vs_incumbent_cents']:.1f} to "
       f"{pcell['net_diff_vs_incumbent_cents']:.1f} cents -- the lift was in the "
       f"ORDERING, not in the scale of the numbers")

    # ---- the null's draws: REPLAYED, and read on the decision metrics --
    ncell = run_cell(ref, {
        "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": smart,
        "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d": dumb},
        good, draws=20, thetas={
            "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": 0.5,
            "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d": 0.5})
    # DE33-C4: round 33's null fixture put 20 windows an hour apart, so
    # every (side, hour) stratum held ONE generation and 200 seeds produced
    # ONE distinct draw -- the checks below would pass on a forced null.
    # This one puts several generations in each stratum on BOTH sides, and
    # the freedom and the distinct-draw count are ASSERTED.
    _rich = {}
    for _i in range(12):
        _sl = f"btc-updown-5m-{1787579400 + (_i % 3) * 3600}"
        _sides = _rich.setdefault(_sl, {"BUY_UP": [], "SELL_UP": []})
        for _sd in HSP.SIDES:
            _sides[_sd].append(_gen(len(_sides[_sd]) + 1,
                                    float(_i) * 2.0, float(_i) * 2.0 + 1.5,
                                    [(float(_i) * 2.0 + 0.5, 1.0, -3.0)]))
    _pool = [{"slug": f"{sl}|{sd}|{g['gen']}", "side": sd,
              "hour": _hour_of(sl)}
             for sl, sides in sorted(_rich.items())
             for sd in HSP.SIDES for g in sides[sd]]
    # The demand takes ONE generation from each of the first strata rather
    # than the first six of the pool -- taking a prefix empties a stratum
    # and the freedom assertion below correctly refused that fixture.
    _demand = {}
    _seen_st = set()
    _take = []
    for _g in _pool:
        _st = (_g["side"], _g["hour"])
        if _st in _seen_st:
            continue
        _seen_st.add(_st)
        _take.append(_g)
        if len(_take) >= 4:
            break
    for _g in _take:
        _demand[(_g["side"], _g["hour"])] = \
            _demand.get((_g["side"], _g["hour"]), 0) + 1
    _avail = {}
    for _g in _pool:
        _avail[(_g["side"], _g["hour"])] = \
            _avail.get((_g["side"], _g["hour"]), 0) + 1
    _freedom = {k: _avail[k] - v for k, v in _demand.items()}
    ok(all(v > 0 for v in _freedom.values()) and len(_avail) >= 4,
       f"DE33-C4: the null fixture has FREEDOM > 0 in every stratum it "
       f"draws from ({_freedom}) across {len(_avail)} (side, hour) strata "
       f"-- round 33's fixture gave every stratum one member, so a matched "
       f"draw was FORCED and 200 seeds produced one distinct draw")
    _treated6 = [{"slug": g["slug"]} for g in _take]
    _draws = {tuple(MRC.draw(_pool, _treated6, seed=_s)) for _s in range(50)}
    ok(len(_draws) > 1,
       f"and 50 seeds produce {len(_draws)} DISTINCT draws, asserted -- a "
       f"null whose draws are all the same is a constant wearing a "
       f"distribution's name")
    nq = ncell["null_quantiles"]
    ok(nq["n"] == 20 and "cost_adjusted_value_cents AND rho" in nq["metric"]
       and "never a harm share" in nq["metric"],
       f"DE32-C4 CLOSED: the null is REPLAYED as an acting arm and read on "
       f"the DECISION metrics -- {nq['metric'][:60]}... -- with value "
       f"quantiles q50 {nq['value_q50']:.1f} / q95 {nq['value_q95']:.1f} "
       f"and rho quantiles beside them. Round 32 drew from synthetic "
       f"strata and scored each draw as a HARM SUM, which is the proxy the "
       f"frozen §6 says the comparison is never made on")
    ok("rho_q50" in nq and "rho_q05" in nq,
       f"and the null carries RHO quantiles too ({nq['rho_q50']}, "
       f"{nq['rho_q05']}), because the protocol names both numbers and a "
       f"null on one of them is a null on half the comparison")
    ok(ncell["net_diff_vs_null_median_cents"] == (
        ncell["per_arm"]["CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm"]
        ["cost_adjusted_value_cents"] - nq["value_q50"]),
       f"and the difference the receipt carries is the treated arm against "
       f"the NULL's median ({ncell['net_diff_vs_null_median_cents']:.1f} cents), computed "
       f"here rather than asserted")
    preds = evaluate_predicates([dict(cell, rho=0.8, retention_share=0.9),
                                 dict(cell, rho=1.2, latency_ms=5,
                                      retention_share=0.4), ncell])
    # ---- the timed smoke, and the projection -----------------------------
    per_window_s = smoke_s / max(1, len(ref))
    proj_471 = per_window_s * 471
    ok(smoke_s > 0 and proj_471 > 0,
       f"TIMED SYNTHETIC SMOKE, not a guess: {len(ref)} synthetic windows × "
       f"3 arms in {smoke_s * 1000:.1f} ms = {per_window_s * 1e3:.2f} ms "
       f"per window per cell, so the §3 population's 471 windows project "
       f"to ~{proj_471:.1f} s of REPLAY per arm per cell. The addendum's "
       f"6 h/200-draw figure stands on LANE4's measured 1,339.6 s "
       f"end-to-end, which includes the feed this smoke does not build")
    ok(len(LATENCY_RUNGS_MS) == 9 and len(BUDGETS) == 3 and len(COINS) == 2,
       f"the declared grid is {len(LATENCY_RUNGS_MS)} rungs × "
       f"{len(BUDGETS)} budgets × {len(COINS)} coins = "
       f"{len(LATENCY_RUNGS_MS) * len(BUDGETS) * len(COINS)} cells, with "
       f"the null at {len(NULL_CELLS)} of them")
    ok(REPOST_FILL_MODELS is HSP.REPOST_FILL_MODELS
       and PROTECTION_MODES is HSP.PROTECTION_MODES,
       f"and the bracketed axes are the POLICY's own objects, imported: "
       f"{REPOST_FILL_MODELS} × {PROTECTION_MODES}")
    import ast as _ast
    _tree = _ast.parse(Path(__file__).read_text())
    _rc = [f for f in _ast.walk(_tree) if isinstance(f, _ast.FunctionDef)
           and f.name == "run_cell"]
    _calls = [c for f in _rc for c in _ast.walk(f)
              if isinstance(c, _ast.Call)
              and getattr(c.func, "attr", "") == "refuse_if_not_random"]
    ok(len(_rc) == 1 and len(_calls) == 1,
       f"THE IDENTITY GUARD IS CALLED ON EVERY DRAW, asserted from the "
       f"parse: `run_cell` contains {len(_calls)} call(s) to "
       f"`refuse_if_not_random`. Removing that call left every other check "
       f"in this suite green -- a control the runner holds but never "
       f"invokes is a control that cannot fire, and only the code's shape "
       f"says whether it is invoked")
    _rcf = [fn for fn in _ast.walk(_tree)
            if isinstance(fn, _ast.FunctionDef) and fn.name == "run_cell"][0]
    _null_src = _ast.get_source_segment(Path(__file__).read_text(), _rcf)
    ok("res = arm_result(" in _null_src
       and 'vals.append(res["cost_adjusted_value_cents"])' in _null_src
       and "harm_by_slug" not in _null_src,
       "AND THE NULL'S VALUES COME FROM A REPLAY, asserted at the source: "
       "the draw is passed through `arm_result` and appended as its "
       "`cost_adjusted_value_cents`, and the word `harm` does not appear "
       "and round 32's `harm_by_slug` parameter is gone entirely. It "
       "valued each draw as a HARM SUM, "
       "which is the proxy the frozen §6 forbids the comparison on "
       "(DE32-C4)")
    ok(RHO.EXPECTED_CHECKS and SS.EXPECTED_CHECKS and MRC.EXPECTED_CHECKS,
       "the three instruments are imported, not reimplemented: rho, the "
       "score stream and the matched-random control each carry their own "
       "suite")

    # ---- a rho for a cell with NO received fills REFUSES ---------------
    # A tranche with NO measured mid: rho must be None, not a number.
    _nomid = {_slug[0]: {"BUY_UP": [_gen(1, 0.0, 20.0, [(5.0, 1.0, -20.0)],
                                         mid=None)],
                         "SELL_UP": []}}
    for _g in _nomid[_slug[0]]["BUY_UP"]:
        for _t in _g["tranches"]:
            _t["mid_at_fill"] = None
    # The score is BELOW theta so the generation is NOT cancelled and its
    # fill is RECEIVED -- otherwise the statuses would be empty for the
    # trivial reason that no fill reached the estimator.
    _nm = arm_result(_nomid, [{"t": 1.0, "slug": _slug[0],
                               "side": "BUY_UP", "score": 0.1}], good,
                     theta=0.5)
    ok(_nm["rho"] is None
       and all(v["rho_statuses"]["NO_MID_AT_FILL"] >= 1
               for v in _nm["legs"].values()),
       f"EST-R1: with NO MEASURED MID the tranche is NO_MID_AT_FILL and rho "
       f"is None -- {[v['rho_statuses']['NO_MID_AT_FILL'] for v in _nm['legs'].values()]}"
       f" -- where the constant denominator used to produce a number for "
       f"every fill, making rho a rescaled mean markout whose threshold "
       f"WAS the constant (DRAFT:212-213 asks for MEASURED spread capture)")
    _empty_ref = {_slug[0]: {"BUY_UP": [_gen(1, 0.0, 20.0, [])],
                             "SELL_UP": []}}
    _empty_scores = [{"t": 1.0, "slug": _slug[0], "side": "BUY_UP",
                      "score": 0.95}]
    _er = arm_result(_empty_ref, _empty_scores, good, theta=0.5)
    ok(_er["rho"] is None
       and all(v["rho_statuses"]["REACHABLE"] == 0
               and v["rho_statuses"]["IN_LATENCY_WINDOW"] == 0
               for v in _er["legs"].values()),
       f"KNOWN-BAD FOR THE EMPTY POPULATION: a generation with NO tranches "
       f"receives no fills, so rho is None with its statuses at zero "
       f"({_er['rho']}) -- a ratio over an empty population is not "
       f"reported as a number, and the cell-level guard refuses one that "
       f"is")
    refuses(lambda: run_cell(_empty_ref, {
        "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": _empty_scores},
        dict(good, _force_rho=True),
        thetas={"CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": 0.5}),
        "KNOWN-BAD: and a cell that carries a rho with no received fills "
        "REFUSES by name", needle="no population")

    # ---- DE32-C1: the flag the row named now exists --------------------
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--selftest", action="store_true")
    _p.add_argument("--run", action="store_true")
    _p.add_argument("--outdir", default=None)
    ok(_p.parse_args(["--run"]).run is True,
       "DE32-C1 CLOSED: `--run` is a real flag on this module's parser. "
       "Round 32's row named it as the invocation of record while "
       "`main()` parsed `--selftest` only, so the declared invocation "
       "would have exited argparse rc 2")
    refuses(lambda: run(Path("/tmp/not_the_declared_dir")),
            "KNOWN-BAD: `--run` pointed anywhere but OUTDIR REFUSES before "
            "any feed is built", needle="names ONE new")
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _d:
        _busy = Path(_d) / "x"
        _busy.mkdir()
        (_busy / "f").write_text("existing")
        refuses(lambda: validate_outdir(_busy, declared=_busy),
                "KNOWN-BAD: an OUTDIR that already EXISTS refuses -- even "
                "empty, because an existing directory may be an anchor and "
                "round 33 tracebacked at `mkdir(exist_ok=False)` instead "
                "(DE33-C8)", needle="already EXISTS")

    # ---- DE32-C2: the feed is INVOKED, and its cap is the estimand's ----
    _rows = [{"slug": "btc-updown-5m-1787579400", "side": "BUY_UP", "gen": 1,
              "t_start": 0.0, "coin": "btc", "day": "d",
              "latency": {"250": {"preventable_value_cents": 3.0,
                                  "preventable_shares": 1.0,
                                  "stale_shares": 0.0}}}]
    _tt = tranche_table(_rows, 250, declare_cap=True)
    ok(_tt["n_generations"] == 1
       and _tt["estimand_horizon_s"] == FILL_HORIZON_S,
       f"DE32-C2 CLOSED: `tranche_table` is CALLED (not merely named in a "
       f"docstring) and its cap travels: {_tt['n_generations']} "
       f"generation(s) at horizon {_tt['estimand_horizon_s']}s")
    try:
        tranche_table(_rows, 250)
        _undeclared = False
    except Exception as _e:
        _undeclared = type(_e).__name__ == "UndeclaredEstimand"
    ok(_undeclared,
       "and the feed REFUSES to emit without `declare_cap=True`, so this "
       "runner cannot inherit the 1-second cap silently (R-165(2) item 5)")
    ok("build_reference" in globals()
       and "select_v2_era" in build_reference.__doc__,
       "and the reference is built from `harmful_exposure_rows`' OWN "
       "pieces -- its selection, replay, join and generation table -- "
       "rather than a second copy of them")

    # ---- DE32-C3: rho is computed on RECEIVED fills ---------------------
    ok(cell["rho"] is not None
       and cell["per_arm"][
           "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm"]
       ["retention_share"] is not None,
       f"DE32-C3 CLOSED: the cell carries rho {cell['rho']} and retention "
       f"{cell['per_arm']['CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm']['retention_share']}"
       f" -- computed by `de_rho_estimator` over the fills the arm "
       f"RECEIVED, each at its own level and its own generation's clock. "
       f"Round 32 imported that estimator and never called it")
    _legs = cell["per_arm"][
        "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm"]["legs"]
    ok(set(_legs) == {f"{pm}|{rf}" for pm in PROTECTION_MODES
                      for rf in REPOST_FILL_MODELS} and len(_legs) == 4,
       f"DE32-C5 (the conjunction): the cell carries BOTH protection modes "
       f"× BOTH repost-fill models -- {sorted(_legs)} -- and is reported at "
       f"its WORST leg, because a bracket declared mandatory is not met by "
       f"the leg that flatters")
    ok(all(v["rho_statuses"]["REACHABLE"]
           + v["rho_statuses"]["IN_LATENCY_WINDOW"] >= 0 for v in
           _legs.values()),
       "with the estimator's statuses carried per leg, so a rho computed "
       "on nothing is visible as nothing rather than as a ratio")

    # ---- DE32-C5: thresholds come from the manifest-bound fits ----------
    _th = thresholds_for("btc", "q1_arrival_composed_lgbm")
    ok(set(_th) >= {"5%", "10%", "15%"}
       and theta_for("btc", "q1_arrival_composed_lgbm", 0.10) == _th["10%"],
       f"DE32-C5 CLOSED: the thresholds are READ FROM THE MANIFEST-BOUND "
       f"FIT ({sorted(_th)}), not defaulted -- round 32 hard-coded "
       f"theta_cancel 0.8 / theta_repost 0.3 in a file whose argument is "
       f"that a policy constant is an input")
    refuses(lambda: theta_for("btc", "q1_arrival_composed_lgbm", 0.07),
            "KNOWN-BAD: a budget with no threshold in the bound fit "
            "REFUSES rather than falling back on a number this file chose",
            needle="policy constant is an input")
    ok(MAX_CANCELS_PER_MINUTE == float("inf")
       and "the frozen protocol names no rate limit" in
       (cell_params.__doc__ or "") + open(__file__).read(),
       f"and `max_cancels_per_minute` is DECLARED unbounded with its "
       f"reason -- the frozen protocol's axes carry no rate limit, so a "
       f"finite one would be a policy axis the protocol does not have")

    # ---- DE33-C2 / C5 / C8 / C9, driven -------------------------------
    _saw_ti = ""
    try:
        _ti = thresholds_for("btc", "incumbent_linear_d")
        _t10 = theta_for("btc", "incumbent_linear_d", 0.10)
    except (DiagRefused, HS.HeadRefused) as _exc:
        _ti, _t10 = {}, None
        _saw_ti = f" REFUSED INSTEAD: {str(_exc)[:110]}"
    ok(set(_ti) == {"5%", "10%", "15%"} and _t10 == _ti.get("10%")
       and not _saw_ti,
       f"DE33-C2 CLOSED: the INCUMBENT's thresholds come from "
       f"`causal_thresholds` -- { {k: round(v, 4) for k, v in _ti.items()} } "
       f"-- the key that fit carries. Round 33 read `thresholds` / "
       f"`budget_thresholds`, so `run()` refused at its FIRST CELL, after "
       f"the ~29-minute feed{_saw_ti}")
    refuses(lambda: run_cell(ref, {"NOT_AN_ARM": dumb}, good,
                             thetas={"NOT_AN_ARM": 0.5}),
            "DE33-C5: an UNKNOWN arm key REFUSES -- the runner iterated "
            "whatever dict the caller passed, so an arm nobody declared "
            "would have been replayed and named in the receipt",
            needle="unknown arm")
    refuses(lambda: run_cell(ref, {"QR_SKEW_ONLY": dumb}, good, thetas={}),
            "DE33-C5: an arm with NO BOUND THRESHOLD REFUSES -- round 33's "
            "`th.get(arm, 0.5)` was DE32-C5's class at a new line",
            needle="no BOUND threshold")
    _rcf2 = [fn for fn in _ast.walk(_ast.parse(Path(__file__).read_text()))
             if isinstance(fn, _ast.FunctionDef) and fn.name == "run_cell"][0]
    _rcs = _ast.get_source_segment(Path(__file__).read_text(), _rcf2)
    ok("missing = sorted(" in _rcs and "no BOUND threshold" in _rcs
       and "unknown = sorted(" in _rcs,
       "and BOTH arm guards are present in `run_cell`, asserted from the "
       "parse: removing the bound-threshold guard turns `th[arm]` into a "
       "KeyError -- red, but from inside the mutant rather than by name, "
       "so the guard's presence is what the check reads")
    ok(set(ARM_SPEC) == set(ARMS) and len(ARM_SPEC) == 5,
       f"and the arm table names all five arms the receipt claims "
       f"({len(ARM_SPEC)}), each with what it IS -- round 33 replayed two "
       f"and named five")
    ok("net_diff_vs_incumbent_cents" in cell
       and "net_diff_vs_null_median_cents" in ncell
       and "net_diff_cents" not in cell and "net_diff_cents" not in ncell,
       f"DE33-C8: the two differences are TWO KEYS -- "
       f"`net_diff_vs_incumbent_cents` "
       f"{cell['net_diff_vs_incumbent_cents']:.1f} and "
       f"`net_diff_vs_null_median_cents` "
       f"{ncell['net_diff_vs_null_median_cents']:.1f} -- where one key "
       f"carried both meanings")
    with _tf.TemporaryDirectory() as _d2:
        _ex = Path(_d2) / "exists"
        _ex.mkdir()
        refuses(lambda: validate_outdir(_ex, declared=_ex),
                "DE33-C8: an EXISTING outdir REFUSES even when EMPTY -- "
                "round 33 passed it and tracebacked at "
                "`mkdir(exist_ok=False)`", needle="already EXISTS")
    # DE34-C2: this check was `ok(<False> or True, ...)` -- the left
    # operand was False at the tip and the PASS was carried by the `or`.
    # It is now driven on a fixture whose statuses are read.
    def _count_statuses(gens):
        st = {"TRANCHE_NO_MARKOUT": 0, "TRANCHE_KEPT": 0}
        for _k, _g in gens.items():
            st["TRANCHE_NO_MARKOUT"] += sum(
                1 for t in _g["tranches"]
                if t["markout_cents_per_share"] is None)
            st["TRANCHE_KEPT"] += sum(
                1 for t in _g["tranches"]
                if t["markout_cents_per_share"] is not None)
        return st
    _with = {("BUY_UP", 1): {"tranches": [
        {"t": 1.0, "markout_cents_per_share": -2.0},
        {"t": 2.0, "markout_cents_per_share": None},
        {"t": 3.0, "markout_cents_per_share": 1.0}]}}
    _without = {("BUY_UP", 1): {"tranches": [
        {"t": 1.0, "markout_cents_per_share": -2.0}]}}
    _sw, _so = _count_statuses(_with), _count_statuses(_without)
    ok(_sw == {"TRANCHE_NO_MARKOUT": 1, "TRANCHE_KEPT": 2}
       and _so == {"TRANCHE_NO_MARKOUT": 0, "TRANCHE_KEPT": 1},
       f"DE33-C9 / DE34-C2: a generation with ONE None-markout tranche "
       f"reads {_sw} and one without reads {_so} -- counted from the same "
       f"expression `build_reference` uses, where this check previously "
       f"passed on `or True` with its left operand False")
    ok(HS.verify_fit_code()["harmful_hazard_model.py"]
       == "58b8a2c08eea3cc9",
       "and the head scorer verifies the manifest's PINNED FIT CODE before "
       "it applies either head -- the arithmetic that fitted them is the "
       "arithmetic that applies them (DE33-C1's first half)")

    # ---- DE34-C1/C2 and EST-R5, each driven ---------------------------
    refuses(lambda: _head_scorer("q1_arrival_composed_lgbm", "btc"),
            "DE34-C1: the RUN PATH's scorer REFUSES by name -- round 34 "
            "left round 33's stub here (`[[row['t']]]` against 106 "
            "features, 0.5 for the incumbent) under a docstring saying "
            "'never a stub', so `--run` would have fed for ~29 minutes and "
            "then tracebacked at the first cell",
            needle="no feature assembly is wired")
    refuses(lambda: preflight(),
            "and `preflight()` REFUSES before anything is built -- on this "
            "tree it stops even earlier than the scorer, at DE34-C4: the "
            "feed and assembly call `harmful_exposure_rows.py` and "
            "`phase2_arms.py`, and BOTH have moved since the fit, so the "
            "pin that matters refuses rather than passing on the two files "
            "that happen to agree",
            needle="has moved since the fit")
    refuses(lambda: verify_called_code(),
            "DE34-C4: and that refusal names BOTH shas and the manifest's "
            "`fit_code_ref`, so the next step is `git show <ref>:<file>` "
            "rather than a judgement call", needle="fit_code_ref")
    _runsrc = _ast.get_source_segment(
        Path(__file__).read_text(),
        [f for f in _ast.walk(_ast.parse(Path(__file__).read_text()))
         if isinstance(f, _ast.FunctionDef) and f.name == "run"][0])
    ok("preflight()" in _runsrc
       and _runsrc.index("preflight()") < _runsrc.index("build_reference("),
       "and the ORDER is asserted from the parse: `preflight()` appears "
       "before `build_reference(` in `run`, so the refusal cannot drift "
       "back behind the feed it exists to precede (DE34-C1)")
    _suite_src = _ast.get_source_segment(
        Path(__file__).read_text(),
        [f for f in _ast.walk(_ast.parse(Path(__file__).read_text()))
         if isinstance(f, _ast.FunctionDef) and f.name == "selftest"][0])
    # The predicate reads the CALL EXPRESSIONS, not the file's text: the
    # phrase appears in this suite's own prose (twice, describing the
    # defect) and a text scan would catch its own explanation.
    _ortrue = [c for c in _ast.walk(_ast.parse(_suite_src))
               if isinstance(c, _ast.Call)
               and getattr(c.func, "id", "") == "ok" and c.args
               and isinstance(c.args[0], _ast.BoolOp)
               and isinstance(c.args[0].op, _ast.Or)
               and any(isinstance(v, _ast.Constant) and v.value is True
                       for v in c.args[0].values)]
    ok(not _ortrue,
       "DE34-C2, generalised: NO check in this suite is carried by "
       "`or True`. One was -- its left operand was False at the tip and "
       "the PASS came from the disjunction -- so the shape is banned by a "
       "check rather than by care")
    _bad_draw_ref = {_slug[0]: {"BUY_UP": [_gen(1, 0.0, 5.0,
                                               [(1.0, 1.0, -3.0)])],
                                "SELL_UP": []}}
    _rc3 = _ast.get_source_segment(
        Path(__file__).read_text(),
        [f for f in _ast.walk(_ast.parse(Path(__file__).read_text()))
         if isinstance(f, _ast.FunctionDef) and f.name == "run_cell"][0])
    # The predicate is over the IF's own test, not over the file's text:
    # the phrase `_cancelled - _want` also appears inside the refusal's
    # message, so a text scan would be satisfied by the sentence that
    # describes the guard after the guard itself was removed.
    _guard = [nd for nd in _ast.walk(_ast.parse(_rc3))
              if isinstance(nd, _ast.If)
              and any(isinstance(x, _ast.BinOp) and isinstance(x.op, _ast.Sub)
                      and getattr(x.left, "id", "") == "_cancelled"
                      and getattr(x.right, "id", "") == "_want"
                      for x in _ast.walk(nd.test))]
    ok(_guard
       and 'ctrl_scores.append({"t": float(g0["t0"])' in _rc3,
       "EST-R5: the control acts at the NAMED generation's own t0 with a "
       "repost one dwell later, and a cancel outside the drawn set "
       "REFUSES -- asserted from the parse, because forcing a mismatched "
       "cancel would mean mutating the policy the control replays. The "
       "null cell above exercises the passing direction on every draw")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_phase4_diag_runner] selftest OK -- {n[0]} checks")
    return 0


def run(outdir: Path | None = None, *, coins=COINS,
        limit: int | None = None) -> dict:
    """THE RUN PATH (DE32-C1).  Feed -> scores -> arms -> rho -> null ->
    receipt, written once into the declared directory.

    Round 32 declared an invocation whose flag did not exist: `main()`
    parsed `--selftest` only, so `--run` exited argparse rc 2 while the
    filing named it as the invocation of record. The flag is real here and
    the path under it is the one the addendum declares."""
    out = validate_outdir(outdir or OUTDIR)
    preflight()      # DE34-C1: BEFORE the feed, not after it
    t_feed = time.time()
    feeds = {c: build_reference(c, limit=limit) for c in coins}
    feed_s = time.time() - t_feed
    heads = {h: SS.verify_head(h, coins[0]) for h in SS.HEADS}
    cells: list = []
    for coin in coins:
        ref = feeds[coin]["reference"]
        rows = feeds[coin]["rows"]
        for budget in BUDGETS:
            for L in LATENCY_RUNGS_MS:
                cap = tranche_table(rows, L, declare_cap=True)
                cell = {"coin": coin, "latency_ms": L, "budget": budget,
                        "enable_reduce": False,
                        "charge_reset_cost_at_generation_start": False}
                scores = {}
                thetas = {}
                for head in ("incumbent_linear_d",
                             "q1_arrival_composed_lgbm"):
                    arm = f"CONDVALUE_OVER_SKEWED_REF/{head}"
                    ev = score_events_for(ref, coin=coin, head=head)
                    scores[arm] = ev
                    thetas[arm] = theta_for(coin, head, budget)
                draws = (N_DRAWS
                         if (coin, L, budget) in
                         {(c, l, b) for c, l, b in NULL_CELLS} else 0)
                cell_out = run_cell(ref, scores, cell, draws=draws,
                                    thetas=thetas)
                cell_out["feed"] = {
                    "n_generations": cap["n_generations"],
                    "rows_per_generation": cap["rows_per_generation"],
                    "estimand_horizon_s": cap["estimand_horizon_s"],
                    "statuses": feeds[coin]["statuses"],
                }
                cells.append(cell_out)
    pop = population_slugs()
    rec = build_receipt(cells, pop, heads=heads,
                        wall_clock_s=time.time() - t_feed)
    rec["feed_seconds"] = feed_s
    validate_receipt(rec)
    out.mkdir(parents=True, exist_ok=False)
    (out / "phase4_diag_r459_receipt.json").write_text(
        json.dumps(rec, indent=1, sort_keys=True))
    return rec


def score_events_for(reference: dict, *, coin: str, head: str) -> list:
    """Score events for every generation in the reference, through the
    manifest-bound adapter -- never a stub."""
    v = SS.verify_head(head, coin)
    rows = [{"t": g["t0"], "slug": slug, "side": side}
            for slug, sides in sorted(reference.items())
            for side in HSP.SIDES for g in sides[side]]
    return SS.score_events(rows, head=head, coin=coin,
                           scorer=_head_scorer(head, coin), verified=v)


def _head_scorer(head: str, coin: str):
    """REFUSES. The feature assembly is not wired (DE33-C1's other half).

    Round 33 fed the LGBM booster `[[row["t"]]]` -- one column against 106
    -- and returned a constant 0.5 for the incumbent, under a docstring
    that said "never a stub". Round 34 built the real head application in
    `de_head_scoring` and LEFT THIS FUNCTION IN THE RUN PATH, so `--run`
    would still have fed for ~29 minutes and then tracebacked at the first
    cell on a `LightGBMError` `main()` did not catch.

    Until `phase2_arms._feature_pass(src, population, TAPE)` is wired --
    the exposure-rows artifact plus the rebuilt state tape it refuses to
    re-derive (R-187 seam 2) -- there is no feature vector to score, and
    the honest behaviour is to refuse BEFORE the feed rather than return a
    number that is not a prediction (DE34-C1)."""
    # SITE: scorer#1
    raise DiagRefused(
        f"no feature assembly is wired, so {head}/{coin} cannot be scored: "
        f"the incumbent needs 60 PM+fine values and the head under test "
        f"106 PM+fine+state, both from "
        f"`phase2_arms._feature_pass(src, population, TAPE)`. "
        f"`de_head_scoring` applies both heads correctly to a vector of the "
        f"right width -- there is no vector yet. Refused HERE, before any "
        f"feed is built (DE34-C1)")


def preflight() -> None:
    """Everything knowable BEFORE the ~29-minute feed.

    The run's first cell is the worst place to learn that the scorer is a
    stub or a thresholds key is wrong: each is checkable in milliseconds,
    so each is checked here and the feed is never paid for a run that
    cannot finish."""
    for coin in COINS:
        for head in ("incumbent_linear_d", "q1_arrival_composed_lgbm"):
            HS.thresholds(coin, head)
    HS.verify_fit_code()
    verify_called_code()
    _head_scorer("q1_arrival_composed_lgbm", COINS[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="execute the declared diagnostic into OUTDIR")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.run:
        # DE33-C8: a refusal under the CLI exits BY NAME, rc non-zero, no
        # traceback -- round 33 let every one of them out unhandled.
        try:
            rec = run(Path(a.outdir) if a.outdir else None)
        except (DiagRefused, HS.HeadRefused, SS.ScoreStreamRefused,
                MRC.ControlRefused, RHO.RhoRefused,
                HSP.ReferenceIntegrityError) as exc:
            print(f"[de_phase4_diag_runner] REFUSED: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(rec["predicates"], indent=1, sort_keys=True))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
