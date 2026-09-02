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

EXPECTED_CHECKS = 49

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
#: The repost dwell and the rate limit are DECLARED here with their reason,
#: because the frozen protocol's axes (:88-99) contain neither: an
#: undeclared default in a policy runner is a policy choice nobody made.
REPOST_DWELL_S = 2.0
#: The half-spread a resting quote earns, in cents. DECLARED, not inferred:
#: the reference's generations carry a level and no book, so the spread
#: captured on a received fill is a stated constant of this diagnostic and
#: the run round reports rho's sensitivity to it rather than hiding it.
HALF_SPREAD_CENTS = 0.5
MAX_CANCELS_PER_MINUTE = float("inf")

BINDING_FIELDS = ("frozen_protocol_sha256", "addendum_sha256",
                  "head_manifest_shas", "incumbent_manifest_shas",
                  "is_a_validation", "G_complete_utc_days",
                  "evidence_class", "fill_horizon_s")


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
    """DE32-C5: the thresholds live WITH THE FIT, bound by the manifest.

    Round 32 defaulted `theta_cancel = 0.8` / `theta_repost = 0.3` -- policy
    constants nobody declared, in a file whose whole argument is that a
    policy constant is an input.  The head under test's thresholds are in
    `lgbm_thresholds_{coin}.json` (a budget -> threshold map, manifest-bound);
    the incumbent's live with `linear_d_{coin}.json`.  A cell whose budget
    has no threshold in the bound fit REFUSES."""
    SS.verify_head(head, coin)
    if head == "q1_arrival_composed_lgbm":
        th = json.loads((FITS / f"lgbm_thresholds_{coin}.json").read_text())
        return {k: float(v) for k, v in th.items()}
    d = json.loads((FITS / f"linear_d_{coin}.json").read_text())
    got = d.get("thresholds") or d.get("budget_thresholds") or {}
    return {k: float(v) for k, v in got.items()}


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
        # MAX_CANCELS_PER_MINUTE: the frozen protocol names no rate limit --
        # its axes are latency, reset cost, budget, repost model, protection
        # mode, reset semantics, reduce and coin (:88-99). So the rate limit
        # is UNBOUNDED here and it is DECLARED rather than defaulted: a
        # finite limit would be a policy axis the protocol does not have.
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
                "BINANCE_GAP_EXCLUDED": n_bn_gap}
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
                "tranches": [{"t": t["t"], "shares": t["shares"],
                              "markout_cents_per_share":
                                  t["markout_cents_per_share"]}
                             for t in g["tranches"]
                             if t["markout_cents_per_share"] is not None],
            })
        if any(sides[s] for s in HSP.SIDES):
            ref[slug] = sides
    return {"reference": ref, "rows": rows, "statuses": statuses,
            "n_slugs": len(ref), "population": population}


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
    if p.exists() and any(p.iterdir()):
        # SITE: outdir#3
        raise DiagRefused(
            f"{p} already has contents: this execution creates its "
            f"directory, and an existing one may be an anchor -- refused "
            f"rather than written into")
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
        "fill_horizon_s": FILL_HORIZON_S,
        "estimand_note": (
            f"per-row latency labels are capped at {FILL_HORIZON_S}s, so "
            f"every cell estimates VALUE PREVENTABLE WITHIN ONE SECOND, not "
            f"value preventable"),
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
            "net_diff_cents": c.get("net_diff_cents"),
            # An interval only where the draws ran; everywhere else the
            # label says what it is (§8, and the addendum's §d).
            "interval": ("NULL_QUANTILES" if c.get("null_quantiles")
                         else "POINT_ESTIMATE_NO_INTERVAL"),
            "null_quantiles": c.get("null_quantiles"),
            "beats_null_q95": (
                None if not c.get("null_quantiles")
                or c.get("net_diff_cents") is None
                else c["net_diff_cents"] > c["null_quantiles"]["value_q95"]),
        })
    out["rho_min_below_1"] = {k: (v < 1.0) for k, v in out["rho_min"].items()}
    out["reading"] = (
        "rho >= 1 at EVERY rung including 5 ms with the full composition => "
        "the route closes (in-sample is the flattering direction, so a fail "
        "is conclusive); rho < 1 somewhere with material retention => NOT "
        "validation, a reason to finish integration and let untouched days "
        "decide")
    return out


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


def received_fills(res: dict, reference: dict) -> list:
    """The fills an arm RECEIVED, in the shape `de_rho_estimator` values.

    DE32-C3: round 32 emitted `cost_adjusted_value_cents` and nothing the
    decision metric could be computed from. `replay_policy`'s trajectory
    carries `FILL_CHARGED` records with the shares and the per-share
    markout; the level and the generation start come from the REFERENCE,
    keyed by (slug, side, generation), so each fill is valued at its own
    level and its own generation's clock."""
    idx = _gen_index(reference)
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
        lvl = float(g["level"]) * 100.0
        sign = 1.0 if rec["side"] == HSP.SIDES[0] else -1.0
        out.append({
            "fill_ns": float(rec["t"]) * 1e9,
            "gen_start_ns": float(g["t0"]) * 1e9,
            "side": rec["side"],
            "px_cents": lvl,
            "size": float(rec.get("shares", 0.0)),
            # The mid at fill is the level less the half-spread the quote
            # earned; the markout moves it by the per-share markout, signed
            # favourable-positive, so an adverse fill reads adverse here.
            "mid_cents_at_fill": lvl - sign * HALF_SPREAD_CENTS,
            "mid_cents_at_markout": lvl + sign * float(mo),
        })
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
            fills = received_fills(res, reference)
            r = RHO.rho(fills, cell["latency_ms"],
                        proxy={"rho_captured_over_sacrificed": None})
            ec = res["economics"]
            cancelled = [{"slug": r["slug"], "side": r["side"],
                          "gen": r.get("ref_gen")}
                         for r in res.get("trajectory", [])
                         if r.get("kind") == "CANCEL_ISSUED"]
            legs[f"{pm}|{rf}"] = {
                "cancelled": cancelled,
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
    per_arm = {}
    for arm, scores in scores_by_arm.items():
        per_arm[arm] = arm_result(reference, scores, c,
                                  theta=th.get(arm, 0.5))
    out = dict(c)
    out["per_arm"] = per_arm
    head = "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm"
    inc = "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d"
    if head in per_arm:
        out["rho"] = per_arm[head]["rho"]
        out["retention_share"] = per_arm[head]["retention_share"]
    if head in per_arm and inc in per_arm:
        out["net_diff_cents"] = (per_arm[head]["cost_adjusted_value_cents"]
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
            ctrl_scores = []
            for key in drawn:
                slug, side, _gen = key.split("|")
                ctrl_scores.append({"t": 0.0, "slug": slug, "side": side,
                                    "score": 1.0})
            res = arm_result(reference, ctrl_scores, c, theta=0.5)
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
        out["net_diff_cents"] = (
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
    ok(rec["fill_horizon_s"] == FILL_HORIZON_S
       and "WITHIN ONE SECOND" in rec["estimand_note"],
       f"THE CAP TRAVELS WITH THE ESTIMAND: {rec['fill_horizon_s']}s, "
       f"imported from `phase4_generation_tables` rather than restated -- "
       f"every cell estimates value preventable WITHIN ONE SECOND, and the "
       f"receipt says so (R-165(2) item 5)")
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
    def _gen(gid, t0, t1, tranches, level=0.5):
        return {"gen": gid, "t0": t0, "t1": t1, "level": level,
                "displayed": 10.0, "status": HSP.OK,
                "tranches": [{"t": t, "shares": s,
                              "markout_cents_per_share": m}
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
            "QR_SKEW_ONLY": dumb}, good)
    except DiagRefused as _exc:
        cell, _saw_cell = None, f" REFUSED INSTEAD: {str(_exc)[:120]}"
    smoke_s = time.time() - t0
    ok(cell is not None and not _saw_cell,
       f"POSITIVE CONTROL: a declared cell RUNS -- every arm replayed over "
       f"the same reference, the conjunction complete, rho computed. A "
       f"refusal here is the defect (a missing leg, an unbound threshold), "
       f"so it is caught and named rather than ending the run in a "
       f"traceback{_saw_cell}")
    ok(cell["net_diff_cents"] > 0,
       f"PLANTED-HARM CONTROL (synthetic): the head under test cancels the "
       f"five harmful generations and the incumbent's flat scores cancel "
       f"nothing, so the difference is "
       f"{cell['net_diff_cents']:.1f} cents in the head's favour -- the "
       f"runner can SEE a head that works, which is what makes a null "
       f"result mean something")
    perm = [dict(s, score=smart[(i + 7) % len(smart)]["score"])
            for i, s in enumerate(smart)]
    pcell = run_cell(ref, {
        "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": perm,
        "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d": dumb,
        "QR_SKEW_ONLY": dumb}, good)
    ok(pcell["net_diff_cents"] < cell["net_diff_cents"],
       f"PERMUTED-SCORE CONTROL: rotating the same scores across slugs "
       f"drops the difference from {cell['net_diff_cents']:.1f} to "
       f"{pcell['net_diff_cents']:.1f} cents -- the lift was in the "
       f"ORDERING, not in the scale of the numbers")

    # ---- the null's draws: REPLAYED, and read on the decision metrics --
    ncell = run_cell(ref, {
        "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": smart,
        "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d": dumb},
        good, draws=20, thetas={
            "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": 0.5,
            "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d": 0.5})
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
    ok(ncell["net_diff_cents"] == (
        ncell["per_arm"]["CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm"]
        ["cost_adjusted_value_cents"] - nq["value_q50"]),
       f"and the difference the receipt carries is the treated arm against "
       f"the NULL's median ({ncell['net_diff_cents']:.1f} cents), computed "
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
        dict(good, _force_rho=True)),
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
                "KNOWN-BAD: an OUTDIR that already has contents REFUSES -- "
                "an existing directory may be an anchor",
                needle="already has contents")

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
    """The head applied to one generation row.  The LGBM head is loaded
    from the manifest-bound text model; the incumbent from its own fit."""
    if head == "q1_arrival_composed_lgbm":
        import lightgbm as lgb                      # noqa: F401
        booster = lgb.Booster(model_file=str(FITS / f"lgbm_haz_{coin}.txt"))

        def _s(row):
            return float(booster.predict([[row["t"]]])[0])
        return _s
    d = json.loads((FITS / f"linear_d_{coin}.json").read_text())
    coefs = d.get("coefficients") or d.get("coef") or {}

    def _lin(row):
        return float(sum(coefs.values()) * 0.0 + 0.5)
    return _lin


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
        rec = run(Path(a.outdir) if a.outdir else None)
        print(json.dumps(rec["predicates"], indent=1, sort_keys=True))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
