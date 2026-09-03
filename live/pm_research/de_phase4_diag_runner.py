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

EXPECTED_CHECKS = 154

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
#: EST-R3: BOTH of these are numbers I chose. `validate_params` refuses
#: `theta_repost >= theta_cancel`, so some hysteresis must exist for the
#: policy to load, and the TODO requires a DECLARED dwell without fixing
#: one. They are proposed to the USER in the v2 DRAFT with sensitivity
#: pairs that select neither; until that ruling they keep these values and
#: nothing in the code cites the draft as authority.
REPOST_DWELL_S = 2.0
#: EST-R4 / DRAFT:71 (row 8 of the §2 parameter table, NOT :88-99 -- the
#: axes table is :99-108): the rate limit is NAMED by the frozen protocol
#: and asks for a per-cell declaration. `inf` is that declaration -- a
#: declared value, not an absent one -- and the identity
#: `requested = passed + suppressed` is reported per arm beside it.
MAX_CANCELS_PER_MINUTE = float("inf")

#: DE34-R7 (RULED) / DE35-R1: THE PIN IS COMPUTED, NOT LISTED, AND IT
#: REPORTS A STATUS RATHER THAN BLOCKING ON A FILE THE RUN NEVER EXECUTES.
#:
#: Round 35 listed two files and refused when either had moved. One of
#: them -- `phase2_arms.py` -- is not in this runner's import closure at
#: all, so the pin blocked the run on code the diagnostic does not run;
#: and a FILE sha cannot say that nine of ten called functions are
#: byte-identical, which is the fact that decides whether the difference
#: reaches this population. The called set is now COMPUTED from the import
#: closure, the comparison is per FUNCTION against the fit-commit bytes,
#: and the residue travels as a verdict:
#:
#:   IDENTICAL          -- every called function's AST dump matches the fit
#:   ADDITIVE_DECLARED  -- some differ, named, with why the run's path is
#:                         unaffected; the run PROCEEDS against the tip
#:   BLOCKING           -- a called function differs in a way that reaches
#:                         the number; the ONLY verdict that refuses
#:   NOT_CALLED         -- pinned by the manifest, never imported here
#:
#: Running the fit-commit bytes instead would reinstate the silent-empty
#: defect `851edaf` exists to remove and needs a materialised pinned import
#: path this runner does not have (R-473).
PIN_VERDICTS = ("IDENTICAL", "ADDITIVE_DECLARED", "BLOCKING", "NOT_CALLED")

#: Differences the run has looked at and declared additive, with the reason
#: each cannot reach the number. A function NOT in this map that differs is
#: BLOCKING -- silence is never the additive answer.
#:
#: DE37-C2/R1 -- WHAT THIS MAP ASSERTS, AND WHAT ROUND 37's VERSION DID
#: NOT. Each entry carries TWO LITERAL AST shas: the function's sha at the
#: fit commit, and its sha at the tip where the declaration was WRITTEN
#: (`cb8aab5`, computed once and typed here). `pin_statuses` compares the
#: file it finds against BOTH literals, so an edit to a declared function
#: re-opens it to BLOCKING.
#:
#: Round 37 computed `sha_at_declaring_tip` at import from the current
#: file -- the same bytes the comparison then read -- so `want == now` by
#: construction and the three declared functions were a PERMANENT
#: EXEMPTION: driven twice (coordinator and reviewer), an edited
#: `select_v2_era` still read ADDITIVE_DECLARED and `verify_called_code()`
#: PROCEEDED, while an edited UNDECLARED function (`join_fills`) BLOCKED.
#: A sha computed from the artifact it is meant to pin is not a pin.
#:
#: DE38 §2(iii): each entry also names the COMMIT THAT CHANGED THE
#: FUNCTION, and that claim is CHECKED -- at `changed_at` the function's
#: AST sha must equal `sha_at_declaring_tip`, and at `changed_at^` it must
#: equal `sha_at_fit`. Nothing can pin a REASON's prose, but this pins the
#: fact the prose is about: a re-declaration that moves the shas while
#: keeping the old justification fails here, because the old commit no
#: longer produces the new bytes.
#:
#: `sha_at_fit: None` is not a missing value -- it is the declared fact
#: that the function is ABSENT from the fit bytes (new at 851edaf), which
#: is what two of these three declarations are about.
DECLARED_ADDITIVE = {
    ("harmful_exposure_rows.py", "select_v2_era"): {
        "changed_at": "851edaf",
        "sha_at_fit": "e97a6662273d8abc",
        "sha_at_declaring_tip": "3b34bdc86b1056ca",
        "reason":
            "the `era` keyword now defaults to `fi.ERA` via "
            "`_era_or_refuse` (the value this "
            "population is selected under anyway) and an empty selection "
            "REFUSES instead of returning nothing -- both make the same "
            "selection for this population and turn a silent empty into a "
            "refusal, which is the direction the fit-commit bytes lack"},
    ("harmful_exposure_rows.py", "_era_or_refuse"): {
        "changed_at": "851edaf",
        "sha_at_fit": None,
        "sha_at_declaring_tip": "830c4fa88ba44280",
        "reason":
            "NEW at 851edaf (absent from the fit bytes): it names the era "
            "from `fi.ERA` -- the value this population is already selected "
            "under -- and routes the empty case to the refusal below. It "
            "adds a refusal and changes no selection that is non-empty, so "
            "for this population the selected set is the fit's"},
    ("harmful_exposure_rows.py", "_refuse_empty_selection"): {
        "changed_at": "851edaf",
        "sha_at_fit": None,
        "sha_at_declaring_tip": "a6cfb900e1ced0b8",
        "reason":
            "NEW at 851edaf: it exists only to RAISE on an empty selection, "
            "so it cannot alter a selection that is non-empty -- and the "
            "population this diagnostic runs on is 471 windows, measured"},
}

#: NOT A DECLARATION.  Round 43's wiring reached `_stream_tape_rows` for
#: the first time and the pin went BLOCKING on it; round 43 then WROTE a
#: declaration and named the doubt.  R-496's dispatch ruled the doubt:
#: admitting a fit-vs-tip drift is an ADMISSIBILITY call and admissibility
#: is the USER's (rule 14), so the declaration is WITHDRAWN and the pin
#: stays BLOCKING.  What lives here instead is the FACT SHEET the ruling
#: needs, and not one field of it is consulted by `pin_statuses` -- a
#: record that the pin read would be a declaration wearing another name.
#: Every claim in it is COMPUTED by `stream_tape_rows_drift()` (rule 10);
#: the two strings below are the CANDIDATES that function verifies, not
#: assertions it repeats.
UNDECLARED_DRIFT = {
    "file": "phase2_arms.py",
    "function": "_stream_tape_rows",
    "candidate_changed_at": "2e1204f",
    "candidate_commit_subject": "BE: T2 fail-open readers",
    "routed_to": "USER",
    "question": (
        "the function differs from the fit-commit bytes and the pin "
        "therefore BLOCKS the run. Whether that difference is ADMISSIBLE "
        "for a diagnostic that must score what the heads were fitted on "
        "is not a seat's call: the facts below are computable and are "
        "computed, the judgement is the USER's"),
    "what_a_grant_would_mean": (
        "an entry in DECLARED_ADDITIVE for (phase2_arms.py, "
        "_stream_tape_rows), after which `verify_called_code()` proceeds "
        "and the run's only remaining gate is the execution act itself"),
}

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


def _direct_imports(src: str) -> set:
    import ast as _a
    out = set()
    for nd in _a.walk(_a.parse(src)):
        if isinstance(nd, _a.Import):
            out.update(al.name.split(".")[0] for al in nd.names)
        elif isinstance(nd, _a.ImportFrom) and nd.module:
            out.add(nd.module.split(".")[0])
    return out


def _ast_sha(dump: str | None) -> str | None:
    return None if dump is None else hashlib.sha256(
        dump.encode()).hexdigest()[:16]


def import_closure(mod_path: Path | None = None) -> set:
    """The modules this runner actually imports, from its own parse --
    including the lazy imports inside functions, which is where
    `harmful_exposure_rows` lives (`build_reference`)."""
    here = (mod_path or Path(__file__)).resolve().parent
    seen, stack = set(), list(_direct_imports(
        (mod_path or Path(__file__)).read_text()))
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        f = here / f"{m}.py"
        if f.exists():                 # FIRST-PARTY only: walk it further
            stack.extend(_direct_imports(f.read_text()) - seen)
    return seen


def called_functions(src: str, entries: set) -> set:
    """The functions REACHABLE from `entries` inside one module's source.

    The ruling says compare the CALLED functions, and "called" is
    transitive: `select_v2_era` calls `_refuse_empty_selection`, so a
    change there reaches the run even though the runner never names it.
    Equally, `select_stratified` and `selftest` are in the file and on no
    path from the entries, so a change there does not."""
    import ast as _a
    tree = _a.parse(src)
    bodies = {nd.name: nd for nd in tree.body
              if isinstance(nd, (_a.FunctionDef, _a.AsyncFunctionDef))}
    seen, stack = set(), [e for e in entries if e in bodies]
    while stack:
        fn = stack.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for nd in _a.walk(bodies[fn]):
            if isinstance(nd, _a.Call):
                nm = getattr(nd.func, "id", None) or getattr(
                    nd.func, "attr", None)
                if nm in bodies and nm not in seen:
                    stack.append(nm)
    return seen


def module_entries(mod: str, src: str) -> set:
    """The functions THIS runner calls on `mod`, from its own parse."""
    import ast as _a
    alias = {mod}
    for nd in _a.walk(_a.parse(src)):
        if isinstance(nd, _a.Import):
            for al in nd.names:
                if al.name == mod and al.asname:
                    alias.add(al.asname)
        elif isinstance(nd, _a.ImportFrom) and nd.module == mod:
            pass
    out = set()
    for nd in _a.walk(_a.parse(src)):
        if isinstance(nd, _a.Call) and isinstance(nd.func, _a.Attribute) \
                and getattr(nd.func.value, "id", "") in alias:
            out.add(nd.func.attr)
    return out


def _fn_asts(src: str) -> dict:
    """name -> normalised AST dump for every top-level FUNCTION, plus one
    entry for the module's TOP-LEVEL BODY.

    DE36-C4: a function-level diff cannot see a changed CONSTANT, and
    `MARKOUT_S` or an import swapped at module level changes what every
    function computes. Docstrings are excluded so a comment reflow is not
    a difference."""
    import ast as _a
    tree = _a.parse(src)
    out = {}
    top = []
    for nd in tree.body:
        if isinstance(nd, (_a.FunctionDef, _a.AsyncFunctionDef)):
            out[nd.name] = _a.dump(nd, annotate_fields=True,
                                   include_attributes=False)
        elif isinstance(nd, (_a.Assign, _a.AnnAssign, _a.Import,
                             _a.ImportFrom)):
            top.append(_a.dump(nd, annotate_fields=True,
                               include_attributes=False))
    out["<module top-level>"] = "\n".join(top)
    return out


def _git_show(ref: str, path: str) -> str | None:
    import subprocess
    r = subprocess.run(("git", "show", f"{ref}:{path}"),
                       cwd=str(ROOT), capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def declaration_groups(declared: dict | None = None) -> list:
    """Every declaration group checked AT ITS OWN COMMIT and its parent.

    One row per (`changed_at`, file): the functions declared there, and
    whether each carries the declared TIP sha at that commit and the
    declared FIT sha at its parent.

    `declared` is injectable ONLY so the grouping can be DRIVEN on a
    fixture with TWO groups (DE40-R1: with a single group in the real map,
    "check only the first group" survived the suite -- a loop whose second
    iteration never happens is not a loop anyone has tested). The run
    always passes the real map."""
    declared = DECLARED_ADDITIVE if declared is None else declared
    by_ca: dict = {}
    for k, v in declared.items():
        by_ca.setdefault((v["changed_at"], k[0]), []).append(k)
    out = []
    for (commit, name), keys in sorted(by_ca.items()):
        path = f"live/pm_research/{name}"
        at = _fn_asts(_git_show(commit, path) or "")
        before = _fn_asts(_git_show(f"{commit}^", path) or "")
        bad = []
        for k in sorted(keys):
            if declared[k]["sha_at_declaring_tip"] != _ast_sha(at.get(k[1])):
                bad.append(f"{k[1]}@{commit}")
            if declared[k]["sha_at_fit"] != _ast_sha(before.get(k[1])):
                bad.append(f"{k[1]}@{commit}^")
        out.append({"changed_at": commit, "file": name,
                    "functions": sorted(k[1] for k in keys),
                    "mismatches": bad, "ok": not bad})
    return out


def pin_statuses(here: Path | None = None) -> list:
    """One computed status per manifest-pinned file (DE34-R7).

    `here` is injectable ONLY so the falsifier can drive this function
    against a COPY of the module directory with one function body edited
    (DE37-C2: the known-bad must be an edited function, not a tampered
    in-memory dict). The run never passes it."""
    m = json.loads((FITS / "fit_manifest.json").read_text())
    codes = m.get("fit_code_files") or {}
    ref = m.get("fit_code_ref") or ""
    here = Path(here) if here is not None else Path(__file__).resolve().parent
    called = import_closure()
    out = []
    for name in sorted(codes):
        mod = name[:-3]
        f = here / name
        sha_run = (hashlib.sha256(f.read_bytes()).hexdigest()[:16]
                   if f.exists() else None)
        row = {"path": name, "sha_at_fit": codes[name], "sha_at_run": sha_run,
               "commit": ref, "functions_changed": [], "verdict": None}
        if mod not in called:
            row["verdict"] = "NOT_CALLED"
            out.append(row)
            continue
        if sha_run == codes[name]:
            # The whole file is byte-identical, so the reached set does not
            # change the answer -- and the receipt says WHICH comparison
            # was made, because the two are not the same claim (DE37 item 7).
            row["comparison"] = "whole-file"
            row["reached"] = None
            row["verdict"] = "IDENTICAL"
            out.append(row)
            continue
        fit_src = _git_show(ref, f"live/pm_research/{name}")
        if fit_src is None:
            row["verdict"] = "BLOCKING"
            row["functions_changed"] = ["<fit bytes unavailable at "
                                        f"{ref}>"]
            out.append(row)
            continue
        a, b = _fn_asts(fit_src), _fn_asts(f.read_text())
        entries = module_entries(mod, Path(__file__).read_text())
        reach = (called_functions(f.read_text(), entries)
                 | called_functions(fit_src, entries)
                 | {"<module top-level>"})
        row["entry_points"] = sorted(entries)
        # DE37 item 7: the VERDICT IS A FUNCTION OF THIS SET, so the set
        # travels with it. Wiring the expensive half will reach
        # `tape_index` and `_feature_pass`, which are not in today's; with
        # `reached` in the receipt that re-opening is visible in the diff
        # of two receipts instead of an IDENTICAL that quietly changed
        # meaning.
        row["comparison"] = "per-function over the reached set"
        row["reached"] = sorted(reach)
        row["n_functions_called"] = len(reach)
        changed = sorted(n for n in reach if a.get(n) != b.get(n))
        row["functions_changed"] = changed
        undeclared = []
        for n in changed:
            key = (name, n)
            if key not in DECLARED_ADDITIVE:
                undeclared.append(n)
                continue
            # DE37-C2: the declaration is pinned to WHAT IT DECLARED, and
            # what it declared is TWO LITERALS in this file. If the fit
            # bytes or the file in front of us has moved since, the pass is
            # not inherited. Round 37 recomputed the tip sha from the very
            # file it was checking, so an edit moved the seal with it.
            want = {k: DECLARED_ADDITIVE[key][k]
                    for k in ("sha_at_fit", "sha_at_declaring_tip")}
            now = {"sha_at_fit": _ast_sha(a.get(n)),
                   "sha_at_declaring_tip": _ast_sha(b.get(n))}
            if want != now:
                undeclared.append(f"{n} (declaration stale: declared "
                                  f"{want}, now {now})")
        row["undeclared"] = undeclared
        row["verdict"] = ("IDENTICAL" if not changed else
                          "BLOCKING" if undeclared else "ADDITIVE_DECLARED")
        row["declared"] = {n: DECLARED_ADDITIVE[(name, n)]["reason"]
                           for n in changed if (name, n) in DECLARED_ADDITIVE}
        row["n_functions_in_file"] = len(set(a) | set(b))
        out.append(row)
    return out


def verify_called_code(rows=None, *, here: Path | None = None) -> list:
    """The statuses, refusing ONLY on BLOCKING (DE34-R7 ruled).

    `rows` is injectable so the refusal can be DRIVEN (rule 15): the
    falsifier builds a status row for a synthetic undeclared change and
    asserts this raises by name -- round 36 shipped the refusal without
    one."""
    rows = pin_statuses(here=here) if rows is None else rows
    bad = [r for r in rows if r["verdict"] == "BLOCKING"]
    if bad:
        # SITE: called#1
        raise DiagRefused(
            f"BLOCKING pin status for "
            f"{ {r['path']: r['functions_changed'] for r in bad} }: a "
            f"CALLED function differs from the fit-commit bytes "
            f"({rows[0]['commit']}) in a way nobody has declared additive. "
            f"A file that merely moved is not blocking -- and a file the "
            f"runner never imports is NOT_CALLED, which is what round 35 "
            f"refused on (DE35-R1)")
    return rows


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
                  wall_clock_s: float, pin=None) -> dict:
    """The receipt, with every binding field computed rather than asserted.

    `pin` is injectable FOR THE SUITE ALONE. The pin now BLOCKS (round
    44: `_stream_tape_rows` is undeclared and admissibility is the
    USER's), so on the run path a receipt CANNOT BE BUILT until that is
    ruled -- `verify_called_code()` refuses first, which is the
    artifact-level guard rule 17 asks for: no receipt exists that was
    produced without the code check passing. The suite passes the REAL
    computed rows (`pin_statuses()`), never a fabricated clean set, and
    the run path passes NOTHING -- asserted from this file's own parse
    rather than promised in this docstring."""
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
        # DE37 item 7: the pin's statuses AND the reached set each verdict
        # was computed over.
        "fit_code_pin": verify_called_code() if pin is None else pin,
        # The drift the pin blocks on, computed rather than described --
        # so a receipt produced AFTER a grant still carries what was
        # granted and on what evidence (rule 16).
        "undeclared_drift": stream_tape_rows_drift(),
        "wall_clock_s": wall_clock_s,
        "decides": "nothing -- this is a diagnostic; the reading is the "
                   "addendum's and the decision is the USER's",
    }
    r["predicates"] = evaluate_predicates(cells)
    return r


def validate_receipt(r: dict) -> dict:
    """Refuse a receipt that is missing anything it is bound by."""
    _tainted = {i: c["produced_under_falsifier_input"]
                for i, c in enumerate(r.get("cells") or [])
                if isinstance(c, dict) and c.get("produced_under_falsifier_input")}
    if _tainted:
        # SITE: receipt#4
        raise DiagRefused(
            f"cell(s) {sorted(_tainted)} were produced under a FALSIFIER "
            f"INPUT ({_tainted}): those parameters exist so a refusal can "
            f"be driven, and a receipt built from a cell that took one is "
            f"a receipt about the falsifier. It says so here and is not "
            f"published (DE38-R4)")
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


def _null_status(c: dict) -> str:
    """The cell's null state, derived from WHAT IT REQUESTED (DE40-R2).

    Round 40 read the absence of `null_quantiles`, so "no null was asked
    for" and "a null was asked for and collapsed" were told apart only by
    a second field, and the enumeration was exhaustive only because
    `null#2` refuses a cell that requested draws and accepted too few.
    That refusal stays exactly as it is; this derivation no longer leans
    on it."""
    if "n_draws_requested" not in c:
        # SITE: pred#1
        raise DiagRefused(
            "a cell carries no `n_draws_requested`, so what it ASKED FOR "
            "cannot be read: a null that was never requested and a null "
            "that collapsed would be reported alike, from the absence of "
            "the same field (DE40-R2)")
    # DE41-R1: read with `.get()` BELOW the guard. Indexing here meant the
    # guard's removal died as `KeyError: 'n_draws_requested'` INSIDE this
    # function while its own known-bad was being driven -- red, but at a
    # traceback rather than at the line that names what broke.
    if not c.get("n_draws_requested"):
        return "NO_NULL_REQUESTED"
    if str(c.get("null", "")).startswith("DEGENERATE"):
        return "NULL_COLLAPSED"
    if c.get("null_quantiles"):
        return "NULL_SAMPLED"
    # SITE: pred#2
    raise DiagRefused(
        f"a cell requested {c['n_draws_requested']} draws and carries "
        f"neither quantiles nor a DEGENERATE declaration. `null#2` refuses "
        f"the run when fewer draws are accepted than requested, so this "
        f"state cannot arise from the runner -- and inventing a fourth "
        f"label for it would report a state nobody computed")


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
            # DE39-R1: WHY there is no interval. A cell whose null
            # COLLAPSED and a cell that never ran one both read
            # POINT_ESTIMATE_NO_INTERVAL with `beats_null_q95` None, and
            # only the second is uninformative about the policy: the first
            # is the measurement that every matched draw was the treated
            # arm, which is a finding about the stratum, not an absence.
            "null_status": _null_status(c),
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


#: §5 (gamma), item 1: how many seeds may be spent to obtain the declared
#: number of ACCEPTED draws. Stated, not felt -- see the comment at the
#: attempt loop; exhausting it REFUSES rather than building the null from
#: whichever draws happened to match.
DRAW_ATTEMPT_BUDGET = 20


def _stratum_of(key, gidx) -> tuple:
    """(side, hour) for a `(slug, side, gen)` key -- the frozen strata."""
    return (key[1], _hour_of(key[0]))


def permuted_stream(treated_scores, drawn, theta: float, gidx):
    """The control's stream: the treated arm's, with the ABOVE-THRESHOLD
    score VALUES permuted within (side, hour) so the drawn generations
    carry them.

    Every generation keeps exactly one event at its own t0; the
    per-stratum multiset of scores is unchanged; nothing is invented and
    nothing is dropped. Returns `(stream, ok)` where `ok` is False when a
    stratum cannot be permuted as asked -- and `run_cell` REJECTS and
    redraws on that, which round 37 did not: it bound the flag once and
    read it zero times (DE37-C1).

    DE37-C3 -- THE BELOW VALUES STAY AT THEIR OWN GENERATIONS. Round 37
    sorted them descending and laid them onto the non-drawn keys in stream
    order, so they moved too. That is a SECOND difference between the arms,
    and it is not inert: repost eligibility is "score < theta_repost
    continuously for REPOST_DWELL_S", so a moved below value changes WHEN a
    held side becomes repost-eligible -- §2's number meeting §5's stream.
    Only the swap moves now: a non-drawn generation that carried a below
    value keeps ITS OWN, and the below values displaced by drawn
    generations go to the non-drawn generations that carried above values
    (there are exactly as many of each, by counting). The assignment is by
    sorted key, so it is deterministic and carries no value order."""
    want = {(sl, sd, int(gn)) for sl, sd, gn in
            (k.split("|") for k in drawn)}
    by_st: dict = {}
    for e in treated_scores:
        key = (e["slug"], e["side"], e.get("gen"))
        by_st.setdefault(_stratum_of(key, gidx), []).append(e)
    out, ok = [], True
    for st, events in by_st.items():
        own = {(e["slug"], e["side"], e.get("gen")): float(e["score"])
               for e in events}
        above = sorted((v for v in own.values() if v >= theta), reverse=True)
        keys = [(e["slug"], e["side"], e.get("gen")) for e in events]
        drawn_here = [k for k in keys if k in want]
        rest = [k for k in keys if k not in want]
        if len(drawn_here) != len(above):
            # The draw named a different number of generations in this
            # stratum than there are above-threshold values to give them:
            # a permutation cannot honour it, so the caller redraws. With
            # the demand taken over ABOVE EVENTS this is unreachable for a
            # well-formed reference -- and it is still checked, because
            # "unreachable" is a claim about the caller.
            ok = False
        pairs = list(zip(drawn_here, above))
        # The below values: every non-drawn generation that carried one
        # KEEPS ITS OWN. Only the ones displaced by a drawn generation
        # move, and they move to the non-drawn generations that carried
        # above values -- the swap, and nothing else (DE37-C3).
        _stay = [k for k in rest if own[k] < theta]
        _needs = sorted(k for k in rest if own[k] >= theta)
        _spare = sorted(own[k] for k in drawn_here if own[k] < theta)
        pairs += [(k, own[k]) for k in _stay]
        pairs += list(zip(_needs, _spare))
        # Any leftover key (only when the counts disagree, i.e. ok=False)
        # keeps its own value, so the multiset is still the stream's own.
        placed = {k for k, _ in pairs}
        for e in events:
            k = (e["slug"], e["side"], e.get("gen"))
            if k not in placed:
                pairs.append((k, float(e["score"])))
        byk = {e["slug"] + "|" + e["side"] + "|" + str(e.get("gen")): e
               for e in events}
        for k, v in pairs:
            src = byk[f"{k[0]}|{k[1]}|{k[2]}"]
            out.append(dict(src, score=v))
    out.sort(key=lambda e: e["t"])
    return out, ok


def _realised_by_stratum(arm: dict, gidx) -> dict:
    """Per-stratum realised action count, read AT THE LEG THE CELL IS
    REPORTED AT for both arms -- summing across the four legs of one and
    reading one leg of the other compares two different quantities, which
    is a 100% rejection rate wearing a matching rule's name."""
    out: dict = {}
    for c in arm["legs"][reported_leg(arm)]["cancelled"]:
        st = (c["side"], _hour_of(c["slug"]))
        out[st] = out.get(st, 0) + 1
    return out


def stream_predicates(treated, control, drawn, theta: float, gidx,
                      rc_treated=None, rc_control=None) -> dict:
    """P1-P4 of the ruling, COMPUTED on the two streams a draw produces."""
    def keys(st):
        return sorted((e["slug"], e["side"], e.get("gen")) for e in st)
    kt, kc = keys(treated), keys(control)
    p1 = kt == kc and len(set(kt)) == len(kt) and len(set(kc)) == len(kc)

    def by_st(stream):
        d: dict = {}
        for e in stream:
            k = (e["slug"], e["side"], e.get("gen"))
            d.setdefault(_stratum_of(k, gidx), []).append(float(e["score"]))
        return {k: sorted(v) for k, v in d.items()}
    p2 = by_st(treated) == by_st(control)
    want = {(sl, sd, int(gn)) for sl, sd, gn in
            (k.split("|") for k in drawn)}
    ctrl_above = {(e["slug"], e["side"], e.get("gen")) for e in control
                  if float(e["score"]) >= theta}
    # DE37-R3: the draw's keys must be IN the stream before the equality is
    # asked. Round 37 filtered `want` down to the stream's keys, so a draw
    # naming a generation the stream does not carry passed, and an empty
    # intersection made P3 vacuously true.
    p3 = bool(want) and want <= set(kc) and ctrl_above == want
    p4 = None if (rc_treated is None or rc_control is None) \
        else rc_treated == rc_control
    return {"P1_key_multisets_equal": p1,
            "P2_stratum_score_multisets_equal": p2,
            "P3_drawn_carry_above_and_only_drawn": p3,
            "P4_realised_action_counts_equal": p4}


def run_cell(reference: dict, scores_by_arm: dict, cell: dict, *,
             draws: int = 0, thetas: dict | None = None,
             _known_bad_demand: bool = False,
             _draw_log: list | None = None) -> dict:
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
    # DE40-R2: WHAT THE CELL REQUESTED, recorded by the cell. `null_status`
    # used to be derived from the ABSENCE of quantiles, which is a fact
    # about the output; "no null was asked for" and "a null was asked for
    # and collapsed" are facts about the INPUT, and only `null#2`'s policy
    # (refusing when `accepted < draws`) kept the two derivations
    # agreeing. A cell that records its own request does not depend on
    # that coincidence.
    out["n_draws_requested"] = int(draws)
    # DE38-R4: a cell produced under a FALSIFIER INPUT carries that fact,
    # and `validate_receipt` refuses to publish it. The flags are inert by
    # default and their call sites are asserted from the parse -- this is
    # the belt to that brace: a receipt produced under them says so, or
    # does not exist.
    _fflags = ([("_known_bad_demand" if _known_bad_demand else None),
                ("_draw_log" if _draw_log is not None else None)])
    _fflags = [f for f in _fflags if f]
    if _fflags:
        out["produced_under_falsifier_input"] = _fflags
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
        # DE35-C4: hoisted -- these were rebuilt inside the seed loop.
        gidx = _gen_index(reference)
        treated_scores = list(scores_by_arm[head])
        _nogen = [e for e in treated_scores if e.get("gen") is None]
        if _nogen:
            # SITE: null#3
            raise DiagRefused(
                f"{len(_nogen)} of {len(treated_scores)} score events do "
                f"not name their generation. (gamma) permutes the "
                f"above-threshold VALUES over GENERATIONS within a "
                f"stratum, so an event with no `gen` cannot be placed: "
                f"every event of one (slug, side) collapses to one key and "
                f"the permutation silently becomes the identity. Round 37 "
                f"built its null from exactly such a stream (DE37-C1)")
        # DE38-R1/C3 -- THE POOL IS THE STREAM'S SUPPORT, not the
        # reference's generations. The draw is over above-threshold EVENTS,
        # so a generation the stream does not carry is a key no draw may
        # legally use: drawing it spends budget and is rejected P3, and --
        # worse, and the reason this is not cosmetic -- `_room` is computed
        # against this pool, so `strata_with_room` counted freedom the draw
        # could not use. The freedom statistic and the draw now read the
        # same support.
        pool_by_key: dict = {}
        for e in treated_scores:
            _k = f"{e['slug']}|{e['side']}|{e['gen']}"
            if _k in pool_by_key:
                # SITE: null#4
                raise DiagRefused(
                    f"two score events name the same generation ({_k}): the "
                    f"pool is the stream's support, and a duplicated key "
                    f"would let one generation be drawn twice and weight "
                    f"its value twice in the null")
            pool_by_key[_k] = {"slug": _k, "side": e["side"],
                               "hour": _hour_of(e["slug"])}
        pool = [pool_by_key[k] for k in sorted(pool_by_key)]
        # DE37-C1(a) -- THE DEMAND IS THE ABOVE-THRESHOLD EVENT COUNT, not
        # the action count. (gamma) asks for the above VALUES permuted over
        # ALL above events, acting and non-acting, so the draw must name as
        # many generations per stratum as there are above values to give
        # them: |drawn_here| == |above| by construction.
        #
        # Round 37 demanded on `_treated_actions`. Every action is an above
        # event and not conversely -- the policy is stateful, a HELD side
        # suppresses later crossings -- so in any stratum with a non-acting
        # above event the demand was too small, `permuted_stream` returned
        # `ok=False` with a TRUNCATED-ZIP stream (above values dropped,
        # below values duplicated), and the run replayed it anyway.
        #
        # The ACTION count keeps its own job, where the frozen text puts
        # it: P4, matched AFTER the replay. Two variables, two uses.
        if _known_bad_demand:
            # The falsifier's input, and the ONLY way to get round 37's
            # demand back: it exists so the rejection can be DRIVEN on this
            # path (rule 15). No run passes it -- asserted from the parse.
            demand_events = [{"slug": f"{a['slug']}|{a['side']}|{a['gen']}"}
                             for a in _treated_actions(per_arm[head])]
        else:
            demand_events = [
                {"slug": f"{e['slug']}|{e['side']}|{e.get('gen')}"}
                for e in treated_scores
                if float(e["score"]) >= th[head]]
        treated = demand_events
        vals, rhos = [], []
        # DE31-R2: the null's own population -- how many strata, how many
        # had room, and how many DISTINCT draws the seeds produced. A
        # stratum with no room contributes a point mass and the receipt
        # says so rather than leaving a reader to infer it.
        _strata = {}
        for g in pool:
            _strata[(g["side"], g["hour"])] = \
                _strata.get((g["side"], g["hour"]), 0) + 1
        _dem = {}
        for t0 in treated:
            _sl, _sd, _gn = t0["slug"].split("|")
            _dem[(_sd, _hour_of(_sl))] = _dem.get((_sd, _hour_of(_sl)), 0) + 1
        _room = {k: _strata.get(k, 0) - v for k, v in _dem.items()}
        _seen_draws = set()
        # DE38-C1 rulings (1), (2) and (4): the ACCEPTED set is the null.
        # Everything a reader needs to judge it is accumulated here --
        # which draws were accepted, how many were DISTINCT, and how many
        # were the IDENTITY (the draw naming exactly the above-carrying
        # generations, which under (gamma) is a legal permutation and is
        # therefore admitted, not refused).
        _above_by_st: dict = {}
        for e in treated_scores:
            if float(e["score"]) >= th[head]:
                _st = (e["side"], _hour_of(e["slug"]))
                _above_by_st.setdefault(_st, set()).add(
                    f"{e['slug']}|{e['side']}|{e['gen']}")
        _accepted_draws: list = []
        _acc_by_st: dict = {}
        _identity_all = 0
        _stream_differs = 0
        _treated_map = {(e["slug"], e["side"], e.get("gen")): e["score"]
                        for e in treated_scores}
        _rc_treated = _realised_by_stratum(per_arm[head], gidx)
        attempted = accepted = rejected = 0
        rej_by_stratum: dict = {}
        # DE37-C1(b): every rejection is counted under ITS OWN reason. A
        # null whose rejections are all filed under "action count" hides a
        # stream defect behind a matching statistic.
        rej_by_reason: dict = {"PERM_NOT_OK": 0, "P1": 0, "P2": 0, "P3": 0,
                               "P4": 0}
        _first_rejection = None
        # DRAW_ATTEMPT_BUDGET: how many seeds may be tried to obtain
        # `draws` ACCEPTED draws. 20x the target, and the number is stated
        # rather than felt: a rejection happens when a permutation moves a
        # realised count across strata, which the measured fixture does for
        # a minority of seeds; 20x leaves room for a population where that
        # minority is large while still REFUSING rather than looping if
        # the acceptance rate is so low that the null would be built from
        # a biased subset. Exhausting it is a refusal, not a smaller null.
        for seed in range(draws * DRAW_ATTEMPT_BUDGET):
            if accepted >= draws:
                break
            # The control is an ACTING arm: it cancels what the draw names,
            # and it is read on the same two numbers as every other arm.
            drawn = MRC.draw(pool, treated, seed=seed)
            _seen_draws.add(tuple(drawn))
            # THE IDENTITY GUARD IS RETIRED HERE (DE38-C1 ruling 1),
            # and deliberately NOT re-pointed at the demand.
            #
            # It was written for a control matched on ACTIONS, where a draw
            # equal to the treated arm's actions meant the control did
            # nothing and the cell had to refuse. Under (gamma) a draw is a
            # uniform choice of which |above| generations carry the above
            # values, and the IDENTITY -- the draw that names exactly the
            # above-carrying generations -- is one of the C(N, k)
            # permutations. A permutation null that excludes it is not the
            # permutation distribution. Measured on the C1 fixture over 200
            # seeds: handed the actions the guard fired 0 times (it returns
            # early, because |drawn| = |above| != |actions| whenever a
            # stratum holds a non-acting above event); handed the demand it
            # would have fired 65 times -- refusing exactly the sample
            # points the null must contain.
            #
            # So the identity is ADMITTED and COUNTED: `n_accepted_identity`
            # per stratum below says how much of the accepted set is the
            # treated arm, and an accepted set of one distinct draw makes
            # the cell DEGENERATE rather than a number. The guard's own
            # module keeps it, with its own suite; this call site is gone
            # and the parse certificate that asserted the call is deleted
            # with it -- a check certifying a call that cannot go red reads
            # as coverage and is worse than no check.
            # The control ACTS: a score above any threshold on exactly the
            # generations the draw named, and nothing else.
            # §5 AS RULED = (gamma): the control's stream is the treated
            # arm's with the score VALUES permuted within (side, hour)
            # over ALL above-threshold events -- acting AND non-acting.
            # Every generation keeps exactly ONE event at its own t0, the
            # per-stratum score multiset is unchanged, nothing invented,
            # nothing dropped.
            #
            # Round 36 built the control from the drawn keys alone, so the
            # drawn generation carried TWO events and the others none, and
            # a `zip` dropped an above value (measured: score multiset
            # [0.1, 0.9] against the treated [0.1, 0.8, 0.9]).
            #
            # And the realised cancel set CANNOT be the drawn set: the
            # policy is stateful, a HELD side suppresses later crossings,
            # so a non-acting above event acts the moment the generation
            # that held it stops cancelling. The frozen text asks for
            # matching on ACTION COUNT (DRAFT:147-156), not on identity --
            # so `control#2`, which refused when the cancel set differed,
            # is WITHDRAWN (DE36-R3) and the match is made on the
            # per-stratum REALISED action count, after the replay.
            ctrl_scores, _perm_ok = permuted_stream(
                treated_scores, drawn, th[head], gidx)
            attempted += 1
            # DE37-C1(b) -- P1-P3 ARE COMPUTED HERE, PER DRAW, BEFORE THE
            # REPLAY, and a failure is a rejection under ITS OWN REASON.
            # Round 37 computed them in the selftest only and discarded
            # `ok`, so a stream the code had already judged wrong was
            # replayed and its value entered the null.
            _pred = stream_predicates(treated_scores, ctrl_scores, drawn,
                                      th[head], gidx)
            _why = (["PERM_NOT_OK"] if not _perm_ok else []) + [
                n.split("_")[0] for n in ("P1_key_multisets_equal",
                                          "P2_stratum_score_multisets_equal",
                                          "P3_drawn_carry_above_and_only_drawn")
                if not _pred[n]]
            if _draw_log is not None:
                _draw_log.append({"seed": seed, "reasons": list(_why),
                                  "accepted": not _why, "value": None})
            if _why:
                rejected += 1
                for r in _why:
                    rej_by_reason[r] = rej_by_reason.get(r, 0) + 1
                if _first_rejection is None:
                    _first_rejection = {"seed": seed, "reasons": _why}
                continue
            res = arm_result(reference, ctrl_scores, c, theta=th[head])
            _rc_ctrl = _realised_by_stratum(res, gidx)
            if _draw_log is not None and _rc_ctrl != _rc_treated:
                _draw_log[-1] = {"seed": seed, "reasons": ["P4"],
                                 "accepted": False}
            if _rc_ctrl != _rc_treated:
                # P4: the decision variable did not match. Reject, count
                # per stratum, and redraw -- never keep a control matched
                # on something other than the frozen variable.
                rejected += 1
                rej_by_reason["P4"] = rej_by_reason["P4"] + 1
                if _first_rejection is None:
                    _first_rejection = {"seed": seed, "reasons": ["P4"]}
                for st in set(_rc_ctrl) | set(_rc_treated):
                    if _rc_ctrl.get(st, 0) != _rc_treated.get(st, 0):
                        rej_by_stratum[f"{st[0]}|{st[1]}"] = \
                            rej_by_stratum.get(f"{st[0]}|{st[1]}", 0) + 1
                continue
            accepted += 1
            _accepted_draws.append(tuple(sorted(drawn)))
            _by_st: dict = {}
            for _dk in drawn:
                _sl, _sd, _gn = _dk.split("|")
                _by_st.setdefault((_sd, _hour_of(_sl)), set()).add(_dk)
            for _st, _keys in _by_st.items():
                _e = _acc_by_st.setdefault(
                    _st, {"size": 0, "distinct": set(), "n_identity": 0})
                _e["size"] += 1
                _e["distinct"].add(frozenset(_keys))
                if _keys == _above_by_st.get(_st, set()):
                    _e["n_identity"] += 1
            # The IDENTITY as a property of the WHOLE draw: THE SET OF
            # ABOVE-CARRYING GENERATIONS IS THE TREATED ARM'S. It is SET
            # identity, not stream identity -- the control's stream equals
            # the treated arm's only when the above values also land back
            # on their own generations, which happens exactly when they
            # descend in time (DE39-C1). The set is what a decision reads:
            # with `enable_reduce` False the score is compared against
            # theta_cancel / theta_repost / theta_reduce and nothing else,
            # so WHICH above value lands on WHICH above-carrying generation
            # cannot change a decision -- measured, swapping two above
            # values in time changes no accepted value, no quantile and no
            # difference. With a reduce band enabled the magnitude enters
            # the decision and this must be re-read as a stream property.
            # Summing the per-stratum counts would give draws x strata,
            # which is not a count of anything a reader wants.
            if set(drawn) == set().union(*_above_by_st.values()):
                _identity_all += 1
            # A SECOND, LABELLED STATISTIC -- how far the permutation
            # moved the STREAM, not how far it moved a DECISION. It is
            # informative about the permutation's reach and it is never
            # the assertion: a decision-inert reordering of the above
            # values changes this number and nothing a reader acts on
            # (DE39-C1 ruling ii).
            if {(e["slug"], e["side"], e.get("gen")): e["score"]
                    for e in ctrl_scores} != _treated_map:
                _stream_differs += 1
            if _draw_log is not None:
                _draw_log[-1]["value"] = res["cost_adjusted_value_cents"]
                # DE39-C1 ruling (iii): a null's audit trail names WHAT WAS
                # DRAWN. Without it a reader (or a check) has to re-derive
                # the draw and hope the derivation matches.
                _draw_log[-1]["drawn"] = sorted(drawn)
            vals.append(res["cost_adjusted_value_cents"])
            if res["rho"] is not None:
                rhos.append(res["rho"])
        if accepted < draws:
            # SITE: null#2
            raise DiagRefused(
                f"only {accepted} of {draws} draws matched the treated "
                f"arm's per-stratum realised action count in "
                f"{attempted} attempts ({rejected} rejected: "
                f"{ {k: v for k, v in sorted(rej_by_reason.items()) if v} }"
                f"). A null built "
                f"from the draws that happened to match is matched on "
                f"acceptance, not on the decision variable -- refusing is "
                f"the honest end (DRAW_ATTEMPT_BUDGET = "
                f"{DRAW_ATTEMPT_BUDGET})")
        vals.sort()
        rhos.sort()
        _n_distinct_accepted = len(set(_accepted_draws))
        out["null_population"] = {
            "n_draws_attempted": attempted,
            "n_draws_accepted": accepted,
            "n_rejected_by_stratum": dict(sorted(rej_by_stratum.items())),
            "n_rejected_by_reason": dict(sorted(rej_by_reason.items())),
            "first_rejection": _first_rejection,
            "predicates_per_draw": ["P1", "P2", "P3"],
            "predicate_note": ("P1-P3 are stream properties, computed per "
                               "draw BEFORE the replay; P4 is the decision "
                               "variable, matched AFTER it. A draw failing "
                               "any of them is rejected under that name "
                               "and redrawn (DE37-C1)"),
            "draw_attempt_budget": DRAW_ATTEMPT_BUDGET,
            "n_strata": len(_strata),
            "strata_with_room": sum(1 for v in _room.values() if v > 0),
            "strata_forced": sorted(f"{k[0]}|{k[1]}" for k, v in _room.items()
                                    if v <= 0),
            # DE38-C1 ruling (2): the statistics that describe THE NULL
            # are computed on the ACCEPTED set. `_seen_draws` accumulates
            # every ATTEMPTED draw, so a statistic over it describes the
            # SAMPLER; the accepted set is what the claim rests on (rule
            # 10). Both are reported, each labelled with its population.
            # DE38-R1/C3: the pool the draw and the freedom statistic
            # both read -- the STREAM's support, its size stated so a
            # reader can see it is not the reference's generation count.
            "pool_source": "the score stream's support (one key per event)",
            "pool_size": len(pool),
            "n_distinct_accepted": _n_distinct_accepted,
            "n_distinct_attempted": len(_seen_draws),
            "point_mass_accepted": _n_distinct_accepted == 1,
            # The count a reader wants globally: accepted draws whose
            # control stream IS the treated arm's. Per stratum it is in
            # `accepted_by_stratum` below (ruling 1 asks for both).
            "n_accepted_identity_whole_draw": _identity_all,
            "n_accepted_stream_differs": _stream_differs,
            "stream_differs_note": (
                "counts accepted draws whose STREAM MAP differs from the "
                "treated arm's -- stream maps, NOT decisions. With "
                "`enable_reduce` False a permutation that moves an above "
                "value onto another above-carrying generation changes this "
                "count and changes no decision, no value and no quantile "
                "(measured). Read it as the permutation's reach, never as "
                "evidence that the null differs (DE39-C1)"),
            # DE38-C1 ruling (4): per stratum, BEFORE any §3 number -- how
            # big the accepted set is there, how many DISTINCT draws it
            # holds, how many of them were the identity, and whether it
            # collapsed. The frozen text (DRAFT:147-156) fixes the matching
            # rule and is silent on what to do when the matched set
            # degenerates; this block is what makes that visible.
            "accepted_by_stratum": {
                f"{k[0]}|{k[1]}": {
                    "size": v["size"],
                    "n_distinct": len(v["distinct"]),
                    "n_accepted_identity": v["n_identity"],
                    "collapsed": len(v["distinct"]) <= 1}
                for k, v in sorted(_acc_by_st.items())},
            "note": ("a stratum with no room contributes a POINT MASS: the "
                     "matched draw there is forced, so its contribution is "
                     "a constant and not a sample (DE31-R2). The IDENTITY "
                     "draw is ADMITTED -- it is one of the C(N, k) "
                     "permutations and a null that excludes it is not the "
                     "permutation distribution -- and counted, so a reader "
                     "sees how much of the accepted set is the treated arm "
                     "(DE38-C1)"),
        }
        if _n_distinct_accepted <= 1:
            # DE38-C1 ruling (2), the refusal half. Rule 6 declares >= 200
            # DRAWS, and 200 copies of one draw is one draw. So the cell
            # publishes NO quantiles and NO difference against a median
            # that would be the treated arm's own value: it says the null
            # DEGENERATED and falls back to the point estimate the addendum
            # already declares for cells without one. Weaker than failing
            # the run (rho and retention for this cell are unaffected) and
            # stronger than a "declared point mass", which invites reading
            # a 0.0 difference as a result.
            out["null"] = (f"DEGENERATE (n_distinct_accepted = "
                           f"{_n_distinct_accepted})")
            out["null_degenerate_reason"] = (
                f"the accepted set holds {_n_distinct_accepted} distinct "
                f"draw(s) over {accepted} acceptances "
                f"({_identity_all} of them the IDENTITY draw -- the "
                f"control stream IS the treated arm's), so the null has "
                f"no spread: any "
                f"quantile would be that one draw's value and any "
                f"difference against its median would be an artefact of "
                f"the matching, not a comparison. No `null_quantiles` and "
                f"no `net_diff_vs_null_median_cents` are published for "
                f"this cell (DE38-C1 ruling 2)")
            out["point_estimate_cents"] = \
                per_arm[head]["cost_adjusted_value_cents"]
            out["point_estimate_label"] = (
                "point estimate, labelled -- no interval, because the "
                "cluster (here the accepted null) does not support one")
        else:
            out["null_quantiles"] = {
                "n": len(vals),
                "metric": "cost_adjusted_value_cents AND rho -- the "
                          "DECISION metrics (frozen §6), never a harm share",
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
    # The REAL rows, computed -- not a fabricated clean set. A fixture
    # that supplies what the code under test should produce is the R-229
    # class, and the row this receipt must carry is the BLOCKING one.
    rec = build_receipt([], pop, heads=heads, wall_clock_s=0.0,
                        pin=pin_statuses())
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
    _rp = rec["fit_code_pin"]
    _rper = [r for r in _rp if r["path"] == "harmful_exposure_rows.py"][0]
    _rwhole = [r for r in _rp if r["comparison"] == "whole-file"]
    ok(len(_rp) == 12 and len(_rwhole) == 10
       and _rper["comparison"] == "per-function over the reached set"
       and "select_v2_era" in _rper["reached"]
       and all(r["reached"] is None for r in _rwhole),
       f"AND THE RECEIPT CARRIES THE PIN WITH THE POPULATION EACH VERDICT "
       f"WAS COMPUTED OVER (DE37 item 7): {len(_rp)} rows, {len(_rwhole)} "
       f"compared WHOLE-FILE (`reached` null, because the bytes match and "
       f"the reached set cannot change that answer) and the rest "
       f"per-function over a named set -- `harmful_exposure_rows.py` over "
       f"{len(_rper['reached'])} entries. An IDENTICAL verdict whose "
       f"reached set has changed is a different claim, and two receipts "
       f"now show that in their diff instead of agreeing in silence")
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
    smart = [{"t": 1.0, "slug": _slug[i], "side": "BUY_UP", "gen": 1,
              "score": 0.95 if i < 5 else 0.05} for i in range(20)]
    dumb = [{"t": 1.0, "slug": _slug[i], "side": "BUY_UP", "gen": 1,
             "score": 0.5} for i in range(20)]
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
    # DE38-C1 ruling (3): THE NULL FIXTURE MUST BE ABLE TO PRODUCE A NULL.
    # Round 38's was the 20-slug one below -- one generation per (side,
    # hour) stratum, so the only draw satisfying the match was the IDENTITY
    # and the accepted set was the treated arm, valued 0.0 against itself.
    # This one puts FOUR generations in ONE stratum, two above threshold
    # and two below, two harmful and two benign, so several distinct draws
    # are acceptable and they do not all carry the same value.
    _fslug = [f"btc-updown-5m-{1787579400 + i * 100}" for i in range(4)]
    _free = {_fslug[i]: {"BUY_UP": [_gen(1, 0.0, 20.0,
                                        [(5.0, 1.0, -20.0 if i < 2 else 4.0)])],
                         "SELL_UP": []} for i in range(4)}
    _fsc = [{"t": 0.0, "slug": _fslug[i], "side": "BUY_UP", "gen": 1,
             "score": (0.9, 0.8, 0.2, 0.1)[i]} for i in range(4)]
    # A generation the REFERENCE carries and the STREAM does not: the pool
    # must not contain it (DE38-R1), and `strata_with_room` must not count
    # the freedom it would appear to give.
    _free["btc-updown-5m-1787579800"] = {
        "BUY_UP": [_gen(1, 0.0, 20.0, [(5.0, 1.0, 4.0)])], "SELL_UP": []}
    _fdumb = [dict(e, score=0.5) for e in _fsc]
    _flog: list = []
    _nerr = ""
    try:
        ncell = run_cell(_free, {
        "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": _fsc,
        "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d": _fdumb},
        good, draws=20, thetas={
            "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": 0.5,
                "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d": 0.5},
            _draw_log=_flog)
    except (DiagRefused, MRC.ControlRefused) as _e:
        # A re-added identity guard refuses the IDENTITY draws this null
        # must contain, and the cell dies. That must be a named red here,
        # not a traceback (DE38-C1 ruling 1).
        ncell, _nerr = None, f"{type(_e).__name__}: {str(_e)[:120]}"
    ok(ncell is not None,
       f"THE FREE NULL FIXTURE BUILDS ITS NULL AT ALL -- no per-draw "
       f"refusal ends the cell{(' -- REFUSED INSTEAD: ' + _nerr) if _nerr else ''}. "
       f"With the identity guard back in the loop this is where the run "
       f"stops, because under (gamma) the identity is a legal draw")
    _np_free = (ncell or {}).get("null_population") or {
        "n_distinct_accepted": 0, "n_distinct_attempted": 0,
        "n_accepted_identity_whole_draw": 0, "accepted_by_stratum": {},
        "pool_size": 0}
    # DE39-C1 rulings (ii)+(iii): the property is asserted on the LOGGED
    # ACCEPTED VALUES -- the decision metric -- and the recomputation of
    # the draws is DELETED. It re-derived them from a pool in `_fsc` order
    # against the runner's SORTED pool: coincident here, unasserted in
    # general, and unnecessary now that the log carries both the value and
    # the draw.
    _acc_log = [r for r in _flog if r["accepted"]]
    _tval = ((ncell or {}).get("per_arm", {})
             .get("CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm", {})
             .get("cost_adjusted_value_cents"))
    _val_differs = sum(1 for r in _acc_log if r["value"] != _tval)
    ok(_np_free["n_distinct_accepted"] >= 2 and _val_differs >= 1
       and _np_free["n_accepted_identity_whole_draw"] >= 1
       and _np_free["n_distinct_accepted"]
       <= _np_free["n_distinct_attempted"]
       and all(r.get("drawn") for r in _acc_log),
       f"DE38-C1 ruling (3), ASSERTED ON THE DECISION METRIC (DE39-C1 "
       f"ruling ii): {_val_differs} of {len(_acc_log)} accepted draws "
       f"carry a VALUE different from the treated arm's ({_tval}) -- that "
       f"is what 'the accepted set is a null' means -- across "
       f"{_np_free['n_distinct_accepted']} distinct accepted draws of "
       f"{_np_free['n_distinct_attempted']} attempted, "
       f"{_np_free['n_accepted_identity_whole_draw']} of them the SET "
       f"identity, each logged with what it drew. Round 39 asserted a "
       f"count of differing STREAM MAPS, which a decision-inert "
       f"reordering of the above values moves")
    ok(_np_free["n_accepted_stream_differs"] >= _val_differs
       and "NOT decisions" in _np_free["stream_differs_note"],
       f"and the stream-map count SURVIVES AS A LABELLED STATISTIC -- "
       f"{_np_free['n_accepted_stream_differs']} of {len(_acc_log)} "
       f"accepted draws moved the stream, against {_val_differs} that "
       f"moved the VALUE -- with its note saying it counts stream maps, "
       f"NOT decisions. It is informative about the permutation's reach "
       f"and it is never the assertion")
    # DE39-C1, THE REORDERING DRIVEN: the same fixture with the two above
    # values swapped in time. Nothing a reader acts on may move.
    _swapped = [dict(e, score=(0.8, 0.9, 0.2, 0.1)[i])
                for i, e in enumerate(_fsc)]
    _slog: list = []
    _scell = run_cell(_free, {
        "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": _swapped,
        "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d": _fdumb},
        good, draws=20, thetas={
            "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": 0.5,
            "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d": 0.5},
        _draw_log=_slog)
    _snp = _scell["null_population"]
    _sacc = [r for r in _slog if r["accepted"]]
    _sval_differs = sum(1 for r in _sacc if r["value"] != _tval)
    # DE40-R3: THE INVARIANCE IS ASSERTED OVER THE WHOLE BLOCK, with the
    # one order-dependent statistic EXCLUDED BY NAME -- so a field added
    # to `null_population` tomorrow is covered without editing this check.
    # Round 40 enumerated six fields; the measured invariant is "every
    # field but `n_accepted_stream_differs`".
    _ORDER_DEPENDENT = ("n_accepted_stream_differs",)
    _blk_a = {k: v for k, v in _np_free.items()
              if k not in _ORDER_DEPENDENT}
    _blk_b = {k: v for k, v in _snp.items() if k not in _ORDER_DEPENDENT}
    # DE41-R2: the count is taken AFTER the comparison, and what the line
    # reports is what DIFFERS. Reporting "ALL n of m identical" from the
    # filtered block asserted the invariant on the one path where the
    # message is load-bearing -- the RED one.
    _diff_fields = sorted(set(_blk_a) | set(_blk_b)
                          if _blk_a != _blk_b else set())
    _diff_fields = [k for k in _diff_fields
                    if _blk_a.get(k, object()) != _blk_b.get(k, object())]
    try:
        _pred_a = evaluate_predicates([ncell])["by_cell"][0]
        _pred_b = evaluate_predicates([_scell])["by_cell"][0]
        _prederr = ""
    except DiagRefused as _e:
        # The predicate row is part of the invariant, so a refusal while
        # building it belongs to THIS check rather than to a traceback.
        _pred_a = _pred_b = None
        _prederr = f" -- PREDICATES REFUSED: {str(_e)[:90]}"
    ok(_pred_a is not None and _blk_a == _blk_b and len(_blk_a) >= 20
       and set(_np_free) == set(_snp)
       and _pred_a == _pred_b
       and sorted(r["value"] for r in _sacc)
       == sorted(r["value"] for r in _acc_log)
       and _sval_differs == _val_differs and _sval_differs >= 1,
       f"DE39-C1 / DE40-R3, THE REORDERING DRIVEN OVER THE WHOLE BLOCK: "
       f"swapping the two above values in time leaves "
       f"{len(_diff_fields)} of the {len(_np_free)} `null_population` "
       f"fields DIFFERENT ({_diff_fields or 'none'}) outside the one "
       f"excluded BY NAME, {list(_ORDER_DEPENDENT)}, and the predicate "
       f"row {'differs' if _pred_a != _pred_b else 'identical'}"
       f"{_prederr}. "
       f"So the accepted value "
       f"multiset, the quantiles (q50 "
       f"{_scell['null_quantiles']['value_q50']}), the difference "
       f"({_scell['net_diff_vs_null_median_cents']}), the distinct count "
       f"({_snp['n_distinct_accepted']}) and the SET-identity count "
       f"({_snp['n_accepted_identity_whole_draw']}) are all identical, and "
       f"the count of accepted draws whose VALUE differs is INVARIANT "
       f"under the reordering ({_sval_differs} == {_val_differs} of "
       f"{len(_sacc)}) -- which is the property that makes it the right "
       f"thing to assert. With `enable_reduce` False the score is read "
       f"only "
       f"against the thresholds, so WHICH above value lands on WHICH "
       f"above-carrying generation cannot change a decision -- only the "
       f"SET can")
    ok(_snp["n_accepted_stream_differs"]
       != _np_free["n_accepted_stream_differs"],
       f"and the ONLY number that moves is the labelled stream-map count "
       f"({_np_free['n_accepted_stream_differs']} -> "
       f"{_snp['n_accepted_stream_differs']}) -- which is exactly why "
       f"round 39's assertion rested on it and this round's does not "
       f"(DE39-C1 ruling ii)")
    ok(_np_free["pool_size"] == len(_fsc)
       and sum(len(v[sd]) for v in _free.values() for sd in HSP.SIDES)
       == len(_fsc) + 1,
       f"DE38-R1/C3, DRIVEN: the pool is the STREAM'S SUPPORT -- "
       f"{_np_free['pool_size']} keys for {len(_fsc)} events, while the "
       f"reference carries "
       f"{sum(len(v[sd]) for v in _free.values() for sd in HSP.SIDES)} "
       f"generations. The extra generation has no score event, so no draw "
       f"may name it; at `dfd4c00` it was in the pool, could be drawn "
       f"(spending budget on a P3 rejection) and -- worse -- counted "
       f"toward `strata_with_room`, so the freedom statistic described a "
       f"support the draw could not legally use")
    _accst = _np_free["accepted_by_stratum"]
    ok(len(_accst) == 1 and not list(_accst.values())[0]["collapsed"]
       and list(_accst.values())[0]["n_distinct"] >= 2
       and list(_accst.values())[0]["size"] == 20,
       f"DE38-C1 ruling (4): the accepted set is reported PER STRATUM "
       f"before any §3 number -- {_accst} -- so a reader sees where the "
       f"null had spread and where it collapsed, which is the fact the "
       f"frozen matching rule is silent about")
    # A REAL cell produced under a falsifier input: no draws, so nothing
    # is sampled -- the point is the marking, and that the receipt refuses
    # it (DE38-R4).
    _r4cell = run_cell(ref, {
        "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": smart}, good,
        thetas={"CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": 0.5},
        _draw_log=[])
    ok(_r4cell.get("produced_under_falsifier_input") == ["_draw_log"],
       f"and a cell produced under a falsifier input SAYS SO in the cell "
       f"itself: {_r4cell.get('produced_under_falsifier_input')}")
    refuses(lambda: validate_receipt(dict(rec, cells=list(rec["cells"])
                                          + [_r4cell])),
        f"DE38-R4, DRIVEN ON A REAL CELL: a cell actually produced under "
        f"a FALSIFIER INPUT carries "
        f"{_r4cell.get('produced_under_falsifier_input')} and the receipt "
        f"REFUSES it -- those parameters exist so a refusal can be driven, "
        f"and a receipt built from a cell that took one is a receipt about "
        f"the falsifier", needle="FALSIFIER")
    # THE DEGENERATE CASE, driven on round 38's own null fixture.
    dcell = run_cell(ref, {
        "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": smart,
        "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d": dumb},
        good, draws=20, thetas={
            "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": 0.5,
            "CONDVALUE_OVER_SKEWED_REF/incumbent_linear_d": 0.5})
    _dnp = dcell["null_population"]
    ok(dcell.get("null", "").startswith("DEGENERATE")
       and "null_quantiles" not in dcell
       and "net_diff_vs_null_median_cents" not in dcell
       and dcell.get("point_estimate_cents") is not None
       and _dnp["n_distinct_accepted"] == 1
       and _dnp["n_accepted_identity_whole_draw"]
       == _dnp["n_draws_accepted"],
       f"DE38-C1 ruling (2), DRIVEN on round 38's own null fixture: one "
       f"generation per stratum makes the identity the only acceptable "
       f"draw, so the accepted set holds "
       f"{_dnp['n_distinct_accepted']} distinct draw "
       f"({_dnp['n_accepted_identity_whole_draw']}/"
       f"{_dnp['n_draws_accepted']} of them "
       f"the identity) and the cell publishes `{dcell.get('null')}` with NO "
       f"quantiles and NO difference against a median that would be the "
       f"treated arm's own value -- falling back to the labelled point "
       f"estimate ({dcell.get('point_estimate_cents')} cents). Round 38 "
       f"published q50 and a 0.0 difference here, by construction")
    _dpred = evaluate_predicates([dcell])["by_cell"][0]
    _npred = evaluate_predicates([ncell])["by_cell"][0]
    _0pred = evaluate_predicates([cell])["by_cell"][0]
    ok(dcell.get("n_draws_requested") == 20
       and ncell.get("n_draws_requested") == 20
       and cell.get("n_draws_requested") == 0,
       f"DE40-R2: the cell RECORDS WHAT IT REQUESTED -- "
       f"{dcell.get('n_draws_requested')} "
       f"draws for the degenerate cell, {ncell.get('n_draws_requested')} "
       f"for the sampled one and {cell.get('n_draws_requested')} for the "
       f"cell that "
       f"asked for no null -- so the state below is read off the INPUT "
       f"rather than off the absence of an output")
    refuses(lambda: _null_status({k: v for k, v in dcell.items()
                                  if k != "n_draws_requested"}),
            "KNOWN-BAD: a cell with no `n_draws_requested` REFUSES at "
            "`pred#1` rather than reading the absence of quantiles -- "
            "which is how a never-requested null and a collapsed one came "
            "to be the same row", needle="what it ASKED FOR")
    refuses(lambda: _null_status({"n_draws_requested": 20}),
            "KNOWN-BAD: a cell that requested draws and carries neither "
            "quantiles nor a DEGENERATE declaration REFUSES at `pred#2` "
            "-- `null#2` makes that state unreachable from the runner, and "
            "a fourth label would report a state nobody computed",
            needle="cannot arise from the runner")
    ok(_dpred["null_status"] == "NULL_COLLAPSED"
       and _npred["null_status"] == "NULL_SAMPLED"
       and _0pred["null_status"] == "NO_NULL_REQUESTED"
       and _dpred["interval"] == _0pred["interval"]
       == "POINT_ESTIMATE_NO_INTERVAL",
       f"DE39-R1, DRIVEN on all three states: `null_status` separates a "
       f"null that COLLAPSED ({_dpred['null_status']}) from one that was "
       f"SAMPLED ({_npred['null_status']}) and from a cell that never ran "
       f"one ({_0pred['null_status']}) -- the first two of which are "
       f"different findings and the first and third of which had "
       f"identical rows at `cd93663` (`interval` "
       f"{_dpred['interval']}, `beats_null_q95` None for both). A "
       f"collapsed null is a measurement about the stratum, not an "
       f"absence of measurement")
    ok(_dpred["interval"] == "POINT_ESTIMATE_NO_INTERVAL"
       and _dpred["null_quantiles"] is None
       and _dpred["beats_null_q95"] is None
       and _dpred["net_diff_vs_null_median_cents"] is None,
       f"and the DEGENERATE cell reaches the predicate table as a LABELLED "
       f"POINT ESTIMATE -- interval `{_dpred['interval']}`, "
       f"`beats_null_q95` {_dpred['beats_null_q95']} -- so no reader can "
       f"take a comparison from a cell whose null was the treated arm. "
       f"That is the fallback the addendum already declares for cells "
       f"without an interval, reached here by the cell not having one")
    ok(all(v["collapsed"] for v in _dnp["accepted_by_stratum"].values())
       and _dnp["n_distinct_attempted"] == 1,
       f"and every stratum of that cell is marked `collapsed` "
       f"({_dnp['accepted_by_stratum']}), with the ATTEMPTED distinct "
       f"count ({_dnp['n_distinct_attempted']}) reported beside the "
       f"accepted one -- the sampler's statistic and the null's statistic "
       f"are different quantities and each is labelled with its "
       f"population (rule 10)")
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
    ok(len(_rc) == 1 and len(_calls) == 0,
       f"THE IDENTITY GUARD IS RETIRED FOR (gamma), asserted from the "
       f"parse: `run_cell` contains {len(_calls)} call(s) to "
       f"`refuse_if_not_random` (DE38-C1 ruling 1). The certificate that "
       f"used to assert the call is deleted with it -- handed the actions "
       f"it fired 0/200 seeds (it returns early whenever |drawn| != "
       f"|actions|), and handed the demand it would have refused the 65 "
       f"IDENTITY draws the permutation null must contain. This check is "
       f"the retirement's own falsifier: re-adding the call turns it red")
    _rcf = [fn for fn in _ast.walk(_tree)
            if isinstance(fn, _ast.FunctionDef) and fn.name == "run_cell"][0]
    _args = [a.arg for a in (_rcf.args.args + _rcf.args.kwonlyargs)]
    # IDENTIFIERS ONLY -- names, attributes and parameters. The prose is
    # deliberately not scanned: this function's own note says the null is
    # "never a harm sum", and a check that went red on the sentence
    # explaining the rule would be a check about text again.
    _names = {n.id for n in _ast.walk(_rcf) if isinstance(n, _ast.Name)} | {
        getattr(n, "attr", "") for n in _ast.walk(_rcf)
        if isinstance(n, _ast.Attribute)}
    ok(not [a for a in _args if "harm" in a.lower()]
       and not [n for n in _names if "harm" in str(n).lower()],
       f"and `run_cell` takes NO harm-keyed argument and names nothing "
       f"harm-keyed anywhere in its body -- read from the parse over "
       f"{len(_args)} parameters and {len(_names)} identifiers (prose "
       f"excluded, deliberately). "
       f"Round 32's `harm_by_slug` parameter is gone, and this is the "
       f"check that would notice it coming back")
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
    refuses(lambda: run(Path("/tmp/not_the_declared_dir"),
                        splits=DECLARED_SPLIT_SETS[RULED_SPLIT_SET]),
            "KNOWN-BAD: `--run` pointed anywhere but OUTDIR REFUSES before "
            "any feed is built -- and it refuses at the OUTDIR even with "
            "the ruled split set named, so the two gates are independent",
            needle="names ONE new")
    refuses(lambda: run(None, splits=None),
            "KNOWN-BAD: `--run` at the DECLARED outdir with NO split set "
            "refuses at the split, before the pin and before any feed -- "
            "the ruling exists and is still not supplied on the caller's "
            "behalf", needle="the split set is UNDECLARED")
    refuses(lambda: preflight(
        splits=DECLARED_SPLIT_SETS[RULED_SPLIT_SET]),
        "AND THE HONEST STATE OF THE DIAGNOSTIC, DRIVEN AT `preflight()` "
        "AND DELIBERATELY NOT AT `run()`: with the RULED split set named, "
        "the split gate PASSES and what refuses is the PIN. Round 44 "
        "closed the producer half and the split half; what is left is one "
        "admissibility ruling that belongs to the USER (rule 14). THIS "
        "CHECK USED TO CALL `run()` AND THAT WAS BE12-S1's DEFECT IN A "
        "NEW PLACE: a control whose subject is a GATE, which passes the "
        "day the gate is granted and becomes a ~29-minute feed inside the "
        "selftest. The mutant that re-declares the drift found it",
        needle="BLOCKING pin status")
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
    # DE34-R2: a docstring grep proved nothing about the code. The call
    # is read from the parse instead: `build_reference` must invoke the
    # module's own selection and generation table, not a local copy.
    _brs = _ast.get_source_segment(
        Path(__file__).read_text(),
        [f for f in _ast.walk(_ast.parse(Path(__file__).read_text()))
         if isinstance(f, _ast.FunctionDef) and f.name == "build_reference"][0])
    _her_calls = {nd.func.attr for nd in _ast.walk(_ast.parse(_brs))
                  if isinstance(nd, _ast.Call)
                  and isinstance(nd.func, _ast.Attribute)
                  and getattr(nd.func.value, "id", "") == "HER"}
    ok({"select_v2_era", "replay_with_recorder", "join_fills",
        "generation_table", "label_rows"} <= _her_calls,
       f"DE34-R2: `build_reference` CALLS the module's own pieces "
       f"{sorted(_her_calls)} -- read from the parse, where the old check "
       f"grepped its own docstring for the word `select_v2_era`")

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
    # DE35-C2: this check used to GREP this file for a sentence -- and it
    # passed because the correction quotes that sentence in order to
    # retract it, so the check survived on its own retraction while its
    # message still asserted the retracted claim. It is now a predicate
    # over what `arm_result` PRODUCES.
    _legs_all = [v for arm in cell["per_arm"].values()
                 for v in arm["legs"].values()]
    ok(_legs_all
       and all(v["max_cancels_per_minute"] == MAX_CANCELS_PER_MINUTE
               for v in _legs_all)
       and all(v["rate_identity_holds"] for v in _legs_all),
       f"EST-R4 / DE35-C2: every one of the {len(_legs_all)} arm-legs in "
       f"this cell carries its declared `max_cancels_per_minute` "
       f"({MAX_CANCELS_PER_MINUTE}) and satisfies the frozen identity "
       f"`requested = passed + suppressed` -- read from the OUTPUT, where "
       f"the old check grepped this file for a sentence it had itself "
       f"retracted. `DRAFT:71` names the rate limit and asks for a "
       f"per-cell declaration; the axes table is :99-108")

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
            "DE34-C1: the RUN PATH's scorer REFUSES by name when it is "
            "handed NO assembled scores -- round 44 wired the assembly, "
            "and the one thing this function must never do is invent a "
            "number when the assembly's output did not reach it (round "
            "33's stub: `[[row['t']]]` against 106 features, 0.5 for the "
            "incumbent, ~29 minutes of feed and then a traceback)",
            needle="no assembled scores were supplied")
    _pre = assembly_preconditions()
    ok(_pre["incumbent_width"] == 60 and _pre["lgbm_norm_width"] == 105
       and _pre["lgbm_width"] == 106 and _pre["state_width"] == 45
       and _pre["fragment_bytes"] > 0 and _pre["tape_bytes"] > 0,
       f"the assembly's CHEAP HALF runs in preflight and is MEASURED: "
       f"incumbent {_pre['incumbent_width']}, state {_pre['state_width']}, "
       f"booster {_pre['lgbm_width']} = {_pre['lgbm_norm_width']} + the "
       f"intercept; fragment {_pre['fragment_bytes']:,} B and tape "
       f"{_pre['tape_bytes']:,} B, both matching the fit manifest's own "
       f"paths and byte counts -- so a moved input is caught in "
       f"milliseconds instead of after a 3.2 GB read")
    _sv_fits = globals()["FITS"]
    with _tf.TemporaryDirectory() as _d:
        _man = json.loads((_sv_fits / "fit_manifest.json").read_text())
        _man["tape_bytes"] = int(_man["tape_bytes"]) - 1
        for _f in _sv_fits.iterdir():
            if _f.is_file():
                (Path(_d) / _f.name).write_bytes(_f.read_bytes())
        (Path(_d) / "fit_manifest.json").write_text(json.dumps(_man))
        globals()["FITS"] = Path(_d)
        try:
            refuses(lambda: assembly_preconditions(),
                    "KNOWN-BAD: a tape whose SIZE differs from the fit's "
                    "record REFUSES -- a file that moved after the fit "
                    "yields features the heads were not fitted on, and "
                    "nothing downstream would say so",
                    needle="the file MOVED after the fit")
        finally:
            globals()["FITS"] = _sv_fits
    ok(assembly_preconditions()["tape_bytes"] == _pre["tape_bytes"],
       "POSITIVE CONTROL: the real manifest still answers after that "
       "injection, so the refusal above came from the mutated copy")
    # the per-generation statistic, DRIVEN on a fixture (the expensive
    # pass is what is missing, not this)
    _norms = HS.load_lgbm_normalisers("btc")
    _incm = HS.load_incumbent("btc")
    _nst = _norms["n_raw"] - _incm["_n_features"]
    def _blk(rows):
        return {"PM": [[0.1 * (i + 1)] * 31 for i in range(len(rows))],
                "FN": [[0.2 * (i + 1)] * (_incm["_n_features"] - 31)
                       for i in range(len(rows))],
                "ST": [[0.3 * (i + 1)] * _nst for i in range(len(rows))],
                "kept": rows}
    _fixref = {"s1": {HSP.SIDES[0]: [{"gen": 0, "t0": 100.0},
                                     {"gen": 1, "t0": 400.0}],
                      HSP.SIDES[1]: []}}
    _rows = [{"slug": "s1", "side": HSP.SIDES[0], "gen": 0, "t_start": -6.0},
             {"slug": "s1", "side": HSP.SIDES[0], "gen": 0, "t_start": -3.0},
             {"slug": "s1", "side": HSP.SIDES[0], "gen": 1, "t_start": -6.0}]
    _gs, _gst, _gsp = generation_scores(_blk(_rows), _fixref, coin="btc",
                                        head="incumbent_linear_d")
    _each = [HS.score_incumbent(_incm, HS.compose_head_inputs(
        _blk(_rows)["PM"][i], _blk(_rows)["FN"][i], _blk(_rows)["ST"][i],
        norms=_norms, incumbent_width=_incm["_n_features"],
        lgbm_width=106)["incumbent_linear_d"]) for i in range(3)]
    ok(_gst["SCORED"] == 2 and _gst["NO_ROWS_KEPT"] == 0
       and abs(_gs[("s1", HSP.SIDES[0], 100.0)] - max(_each[0], _each[1])) < 1e-12
       and abs(_gs[("s1", HSP.SIDES[0], 400.0)] - _each[2]) < 1e-12,
       f"DRIVEN: a generation's score is the MAX over its rows "
       f"({_each[0]:.6f}, {_each[1]:.6f} -> "
       f"{_gs[('s1', HSP.SIDES[0], 100.0)]:.6f}), which is the statistic "
       f"`phase2_arms.freeze_thresholds` resolves theta over. A mean or a "
       f"first-row score is compared against a cutoff taken from a "
       f"different distribution and selects the wrong count")
    _gs2, _gst2, _ = generation_scores(_blk(_rows[:1]), _fixref,
                                       coin="btc",
                                       head="incumbent_linear_d")
    ok(_gst2["NO_ROWS_KEPT"] == 1 and _gst2["SCORED"] == 1
       and ("s1", HSP.SIDES[0], 400.0) not in _gs2,
       f"KNOWN-BAD: a generation whose rows the feature pass DROPPED is "
       f"counted as NO_ROWS_KEPT ({_gst2['NO_ROWS_KEPT']}) and is absent "
       f"from the scores -- an exclusion with a status, never a "
       f"generation scored from nothing (rule 4)")
    refuses(lambda: _head_scorer("incumbent_linear_d", "btc", _gs2)(
        {"slug": "s1", "side": HSP.SIDES[0], "t": 400.0}),
        "and the scorer REFUSES that generation if the population was not "
        "filtered first -- the excluded generation must be removed before "
        "scoring, not scored from a miss", needle="no assembled score")
    ok(abs(_head_scorer("incumbent_linear_d", "btc", _gs)(
        {"slug": "s1", "side": HSP.SIDES[0], "t": 100.0})
        - max(_each[0], _each[1])) < 1e-12,
       "POSITIVE CONTROL: given the assembled scores the SAME scorer "
       "returns the generation's number, so the refusal above is about "
       "the miss and not about the scorer being inert")
    refuses(lambda: generation_scores(
        {"PM": _blk(_rows)["PM"][:2], "FN": _blk(_rows)["FN"],
         "ST": _blk(_rows)["ST"], "kept": _rows}, _fixref, coin="btc",
        head="incumbent_linear_d"),
        "KNOWN-BAD: unequal parallel blocks REFUSE -- zipping them at "
        "unequal length pairs one row's features with another row's "
        "identity, silently", needle="parallel")
    refuses(lambda: preflight(),
            "and `preflight()` REFUSES before anything is built -- now at "
            "the PIN, not at the scorer. Round 44 wired the expensive "
            "half, so the scorer is satisfied and what stands in the run's "
            "way is a CALLED function that differs from the fit bytes. "
            "That is an ADMISSIBILITY question and this seat does not "
            "answer it (rule 14)",
            needle="BLOCKING pin status")
    refuses(lambda: preflight(splits=["not_a_split"]),
            "and `preflight(splits=...)` refuses an UNKNOWN split in "
            "milliseconds, before the pin and before any read -- a name "
            "the tape does not carry indexes nothing",
            needle="are not in the tape")
    _pin = pin_statuses()                # the rows, computed
    refuses(lambda: verify_called_code(),
            "DE44: `verify_called_code()` REFUSES BY NAME on the "
            "BLOCKING verdict. Round 43 wrote a declaration for "
            "`_stream_tape_rows` and named its own doubt; R-496's "
            "dispatch ruled the doubt -- the declaration is WITHDRAWN and "
            "the pin stays blocking until the USER grants it",
            needle="_stream_tape_rows")
    _pv = {r["path"]: r["verdict"] for r in _pin}
    _her = [r for r in _pin if r["path"] == "harmful_exposure_rows.py"][0]
    ok(_her["verdict"] == "ADDITIVE_DECLARED"
       and _her["functions_changed"] == ["_era_or_refuse",
                                         "_refuse_empty_selection",
                                         "select_v2_era"]
       and _her["n_functions_called"] >= 17,
       f"DE34-R7 / DE35-R1: THE PIN IS COMPUTED, AND IT NO LONGER BLOCKS. "
       f"The called set is derived from this runner's own import closure "
       f"and the entry points it calls, then closed transitively: "
       f"{_her['n_functions_called']} of {_her['n_functions_in_file']} "
       f"entries in `harmful_exposure_rows.py` (functions plus the "
       f"module's top-level body) are on the run's path, "
       f"and the three that differ from the fit bytes "
       f"({_her['functions_changed']}) are each DECLARED additive with "
       f"their reason -> {_her['verdict']}. The run proceeds against the "
       f"TIP, as R-473 rules")
    _pa = [r for r in _pin if r["path"] == "phase2_arms.py"][0]
    _tc = tape_rows_array_closed()
    ok(_pa["verdict"] == "BLOCKING"
       and _pa["n_functions_called"] >= 5
       and _pa["functions_changed"] == ["_stream_tape_rows"]
       and _pa["undeclared"] == ["_stream_tape_rows"]
       and ("phase2_arms.py", "_stream_tape_rows") not in DECLARED_ADDITIVE,
       f"THE PROPHECY OF ROUND 37 CAME TRUE, AND THE PIN IS STILL HOLDING: "
       f"wiring the expensive half moved `phase2_arms.py` from IDENTICAL "
       f"at 1 reached entry to {_pa['n_functions_called']} "
       f"({_pa['entry_points']} and what they call), and one of them -- "
       f"`_stream_tape_rows` -- DIFFERS from the fit bytes. It is "
       f"{_pa['verdict']} and UNDECLARED ({_pa['undeclared']}) because "
       f"admitting a fit-vs-tip drift is an admissibility call and "
       f"admissibility is the USER's (rule 14). What this round adds is "
       f"not a declaration but the FACTS, each computed")
    _drift = stream_tape_rows_drift()
    ok(_drift["differs"] and _drift["accepting_path_unchanged"]
       and _drift["n_substitutions_that_restore_the_fit"] == 1
       and _drift["enclosing_test"] == "not chunk"
       and _drift["changed_at_verified"]
       and _drift["sha_at_fit"] == "f0741bc4b170fabc"
       and _drift["sha_at_tip"] == "f0b3bccfb8ec5b88"
       and _tc["rows_array_closed"],
       f"DE44, THE FOUR SENTENCES TURNED INTO FOUR PREDICATES: "
       f"`{_drift['function']}` differs "
       f"({_drift['sha_at_fit']} -> {_drift['sha_at_tip']}, verified at "
       f"BOTH sides of {_drift['candidate_changed_at']}); the ACCEPTING "
       f"PATH IS UNCHANGED, established by SUBSTITUTION -- putting a bare "
       f"`return` back where the tip raises makes the whole function's "
       f"AST equal the fit commit's, and exactly "
       f"{_drift['n_substitutions_that_restore_the_fit']} substitution "
       f"does that; the changed statement sits under "
       f"`if {_drift['enclosing_test']}:`, which is EOF; and this tape's "
       f"rows array IS closed ({_tc['tail']!r}, {_tc['bytes']:,} B), so "
       f"the added branch cannot fire for this input. Round 43 asserted "
       f"all four in prose")
    _bad_commit = stream_tape_rows_drift(candidate="669ef72")
    ok(not _bad_commit["changed_at_verified"]
       and _bad_commit["accepting_path_unchanged"],
       f"KNOWN-BAD on the COMMIT: a commit that did not change the "
       f"function reads `changed_at_verified` False "
       f"({_bad_commit['changed_at_verified']}) while the substitution "
       f"clause -- which is about the two ENDPOINTS and not about the "
       f"commit -- stays True. Two claims, two failure modes, and the "
       f"check can tell them apart")
    _pa_src = (Path(__file__).resolve().parent
               / "phase2_arms.py").read_text()
    _needle = "\n            yield obj\n"
    ok(_pa_src.count(_needle) == 1,
       f"and the tamper site for the control below is UNIQUE in "
       f"`phase2_arms.py` ({_pa_src.count(_needle)} occurrence) -- a "
       f"known-bad built by a string replace that silently matches "
       f"nothing is the defect that once reported a clean surface (rule "
       f"15), so the match is asserted before it is used")
    _tampered = _pa_src.replace(
        _needle, "\n            pass\n            yield obj\n", 1)
    _tam = stream_tape_rows_drift(tip_src=_tampered)
    ok(_tampered != _pa_src and not _tam["accepting_path_unchanged"]
       and _tam["n_substitutions_that_restore_the_fit"] == 0,
       f"KNOWN-BAD on the ACCEPTING PATH: a tip carrying ONE extra "
       f"statement on the accepting side (a `pass` before `buf += chunk`) "
       f"reads `accepting_path_unchanged` False with "
       f"{_tam['n_substitutions_that_restore_the_fit']} restoring "
       f"substitutions -- so the clause is a MEASUREMENT of the other "
       f"paths and not a restatement of the one that changed. This is the "
       f"control that decides whether the whole fact sheet is worth "
       f"anything (rule 16)")
    _rep = code_drift_report()
    ok(_rep["run_is_blocked_by_the_pin"] is True
       and _rep["blocking"] == {"phase2_arms.py": ["_stream_tape_rows"]}
       and _rep["undeclared_drift"]["routed_to"] == "USER"
       and _rep["undeclared_drift"]["computed"]["accepting_path_unchanged"],
       f"and the report an operator can read WITHOUT running anything "
       f"(`--pin-report`) carries the verdict DERIVED from "
       f"`pin_statuses()` ({_rep['blocking']}) rather than a literal, so "
       f"a grant changes it on its own and nothing here has to be edited "
       f"to notice")
    # ---- DE44: THE SPLIT IS RULED, AND STILL NOT A DEFAULT ------------
    ok(RULED_SPLIT_SET == "MECHANICS_BOTH_SPLITS"
       and SPLIT_RULING["ruled_by"] == "R-496 (E)"
       and DECLARED_SPLIT_SETS[RULED_SPLIT_SET] == ("score", "train")
       and validate_splits(DECLARED_SPLIT_SETS[RULED_SPLIT_SET])
       == ("score", "train"),
       f"DE44 / R-496 (E): the USER's ruling is RECORDED "
       f"({RULED_SPLIT_SET}, {SPLIT_RULING['ask']}) and the set it names "
       f"validates -- the ADMITTING half of the control, because a guard "
       f"shown only to refuse has not been shown to let the right thing "
       f"through (SEAT_PROTOCOL rule 16)")
    refuses(lambda: validate_splits(None),
            "KNOWN-BAD, THE OTHER DIRECTION: silence still REFUSES BY "
            "NAME even though the ruling exists. A ruling the code "
            "supplies when nobody names it is a ruling nobody has to "
            "read, and the run would then proceed under a population "
            "statement no operator ever typed (rule 14)",
            needle="the split set is UNDECLARED")
    refuses(lambda: validate_splits([]),
            "KNOWN-BAD: an EMPTY set refuses -- it selects no rows, so "
            "every generation would drop and the run would read as a null "
            "result", needle="selects no rows")
    _sig = __import__("inspect").signature(run).parameters["splits"]
    ok(_sig.kind is _sig.KEYWORD_ONLY and _sig.default is _sig.empty
       and _splits_from_cli(None) is None
       and _splits_from_cli("MECHANICS_BOTH_SPLITS") == ("score", "train"),
       f"and `run(splits=...)` is KEYWORD-ONLY WITH NO DEFAULT "
       f"({_sig.default is _sig.empty}) while the CLI translates an "
       f"ABSENT --splits to None rather than to {RULED_SPLIT_SET} -- read "
       f"from the signature and from the function, so a later edit that "
       f"adds a convenience default fails HERE")
    _sb = {("s1", HSP.SIDES[0], 0): "train",
           ("s1", HSP.SIDES[0], 1): "score",
           ("s1", HSP.SIDES[1], 0): "MIXED"}
    _tally = split_tally(list(_sb) + [("s9", HSP.SIDES[0], 7)], _sb)
    ok(_tally == {"train": 1, "score": 1, "MIXED": 1, "UNLABELLED": 1},
       f"DRIVEN: `split_tally` is the whole content of \"labelled per "
       f"cell\" -- {_tally}. MIXED and UNLABELLED are NAMED BUCKETS, "
       f"never folded into a split: a generation whose rows came from "
       f"both splits is not a `train` generation, and one the map does "
       f"not carry is an exclusion with a status (rule 4)")
    _blk3 = _blk(_rows)
    _, _, _sp3 = generation_scores(
        _blk3, _fixref, coin="btc", head="incumbent_linear_d",
        split_of={("s1", HSP.SIDES[0], 0, -6.0): "train",
                  ("s1", HSP.SIDES[0], 0, -3.0): "score",
                  ("s1", HSP.SIDES[0], 1, -6.0): "train"})
    ok(_sp3[("s1", HSP.SIDES[0], 0)] == "MIXED"
       and _sp3[("s1", HSP.SIDES[0], 1)] == "train",
       f"and a generation whose two rows were indexed under DIFFERENT "
       f"splits is labelled {_sp3[('s1', HSP.SIDES[0], 0)]!r}, not "
       f"whichever row came first -- the generation is the unit, so its "
       f"label is a property of all of its rows")
    _, _, _sp4 = generation_scores(_blk3, _fixref, coin="btc",
                                   head="incumbent_linear_d")
    ok(set(_sp4.values()) == {"UNLABELLED"},
       f"KNOWN-BAD: with NO split map every scored generation reads "
       f"UNLABELLED ({sorted(set(_sp4.values()))}) rather than defaulting "
       f"to a split -- an unlabelled cell must be visible in the receipt, "
       f"because under MECHANICS_BOTH_SPLITS the label is the ONLY thing "
       f"distinguishing a fitted generation from an unfitted one")

    # ---- DE44: THE COST, MEASURED RATHER THAN PROJECTED ---------------
    _roots = input_roots()
    ok(_roots["agree"] == (_roots["derived_root"] == _roots["archive_root"])
       and Path(_roots["derived"]).name == "derived"
       and input_roots(archive_repo=_roots["derived_root"])["agree"] is True
       and input_roots(archive_repo="/")["agree"] is False,
       f"DE44: the fit stack reads from TWO ROOTS and the runner computes "
       f"both -- tape/fragment from {_roots['derived_root']} (HARDCODED "
       f"in `phase2_arms`), window archives from "
       f"{_roots['archive_root']} (`__file__`-relative in "
       f"`flow_intensity`). They agree in the main tree and DISAGREE in "
       f"every worktree ({_roots['agree']} here), which is a split-brain "
       f"input that costs a whole tape index before it shows up as "
       f"`no_archive` 100%. DRIVEN BOTH WAYS: pointed at the derived root the "
       f"comparison reads AGREE, pointed at `/` it reads DISAGREE")
    _mat = feature_pass_materialises_whole_file()
    ok(_mat["materialises_whole_file"]
       and _mat["whole_file_loads"] == ["json.loads(src.read_text())"],
       f"and the memory caveat is a PREDICATE over the fit's own parse, "
       f"not a sentence: `_feature_pass` reads its input whole "
       f"({_mat['whole_file_loads']}), so its requirement is a property "
       f"of the full fragment and does not scale down with a row slice")
    _ex = extrapolate(10.0, slice_unit=100, full_unit=250, unit="rows")
    ok(_ex["provenance"] == "EXTRAPOLATION" and _ex["estimate"] == 25.0
       and _ex["factor"] == 2.5
       and "LINEAR IN THE STATED UNIT" in _ex["assumption"],
       f"and every scaled number CARRIES ITS PROVENANCE: {_ex['measured']} "
       f"measured on {_ex['slice_unit']} {_ex['unit']} -> "
       f"{_ex['estimate']} at {_ex['full_unit']}, stamped "
       f"{_ex['provenance']} with the linearity assumption attached. A "
       f"number whose provenance is an extrapolation says so in the row")
    refuses(lambda: extrapolate(1.0, slice_unit=0, full_unit=10,
                                unit="rows"),
            "KNOWN-BAD: an extrapolation from a slice of ZERO refuses -- "
            "a factor with nothing under it is not an estimate",
            needle="division by nothing")
    refuses(lambda: scratch_outdir(OUTDIR),
            "KNOWN-BAD: a measurement outdir that IS the declared "
            "diagnostic OUTDIR refuses -- pre-creating it would turn the "
            "run's already-exists refusal into a false alarm",
            needle="DECLARED diagnostic OUTDIR")
    refuses(lambda: scratch_outdir(ROOT / "data/pm_5min/derived/de44_x"),
            "KNOWN-BAD: a measurement outdir under `data/` refuses -- "
            "this seat is read-only there, and a sliced copy of the tape "
            "landing beside the tape is how a diagnostic input becomes a "
            "production one", needle="READ-ONLY under")
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        _sd = scratch_outdir(Path(_td) / "m")
        ok(_sd.is_dir() and _sd.resolve() != OUTDIR.resolve(),
           f"POSITIVE CONTROL: an ordinary scratch directory is ADMITTED "
           f"and created ({_sd.name}) -- the guard above refuses two "
           f"things and lets the right one through, which is the half a "
           f"guard shown only to refuse never proves")
    import tempfile as _tf2
    with _tf2.TemporaryDirectory() as _td2:
        _f = Path(_td2) / "rows.json"
        _f.write_text(json.dumps({"rows": [{"a": i} for i in range(500)]}))
        _rd = row_density(_f)
        ok(_rd["rows_in_sample"] == 500
           and abs(_rd["estimated_total_rows"] - 500) < 1
           and _rd["basis"].startswith("the ORIGINAL file"),
           f"DRIVEN: `row_density` recovers {_rd['rows_in_sample']} rows "
           f"and estimates {_rd['estimated_total_rows']:.1f} on a file "
           f"whose true count is 500 -- and it reads the ORIGINAL "
           f"artifact rather than the slice this runner writes, because "
           f"`json.dumps` re-serialises and every count taken from a "
           f"slice comes out biased in one direction")
        _f.write_text('{"rows": []}')
        refuses(lambda: row_density(_f),
                "KNOWN-BAD: a file with an EMPTY rows array refuses -- a "
                "density of zero rows per byte makes every extrapolation "
                "infinite, and an infinite estimate reads as a number",
                needle="no complete row decoded")
    # ---- DE44: THE WIRING, ASSERTED FROM THE PARSE --------------------
    # SEAT_PROTOCOL rule 17 names the distinction this check sits on: a
    # control that cannot FAIL is not a control that cannot RUN. `run()`
    # CANNOT be executed while the pin blocks, so the honest instrument
    # for its wiring today is its own parse. It is NOT the seam test --
    # that runs with the ruled execution, on the producer's real rows --
    # and this check exists so the wiring cannot rot in the meantime.
    _runsrc = _ast.get_source_segment(Path(__file__).read_text(),
                                      [nd for nd in _ast.walk(_ast.parse(
                                          Path(__file__).read_text()))
                                       if isinstance(nd, _ast.FunctionDef)
                                       and nd.name == "run"][0])
    _rtree = _ast.parse(_runsrc.replace("\ndef run", "\ndef run"))
    _calls = {}
    for nd in _ast.walk(_rtree):
        if isinstance(nd, _ast.Call):
            nm = getattr(nd.func, "id", None) or getattr(nd.func, "attr", "")
            _calls.setdefault(nm, []).append(
                sorted(k.arg for k in nd.keywords if k.arg))
    _assigns = [_ast.unparse(t) for nd in _ast.walk(_rtree)
                if isinstance(nd, _ast.Assign) for t in nd.targets]
    ok("assemble_gen_scores" in _calls
       and any("gen_scores" in kw for kw in _calls.get("score_events_for", []))
       and any("splits" in kw for kw in _calls.get("assemble_gen_scores", []))
       and "cell_out['splits']" in _assigns
       and _calls.get("build_receipt")
       and all("pin" not in kw for kw in _calls["build_receipt"]),
       f"DE44, THE WIRING FROM THE PARSE (rule 17): `run()` calls "
       f"`assemble_gen_scores(splits=...)`, hands its output to "
       f"`score_events_for(gen_scores=...)` "
       f"({_calls.get('score_events_for')}), assigns `cell_out['splits']`, "
       f"and calls `build_receipt` with NO `pin=` "
       f"({_calls['build_receipt']}) -- so the suite's injectable pin "
       f"cannot leak onto the run path. Round 43's expensive half existed "
       f"as functions with no call site, which is the defect this check "
       f"is about")
    _first = {}
    for nd in _ast.walk(_rtree):
        if isinstance(nd, _ast.Call):
            nm = getattr(nd.func, "id", None) or getattr(nd.func, "attr", "")
            _first.setdefault(nm, nd.lineno)
            _first[nm] = min(_first[nm], nd.lineno)
    ok(_first["validate_outdir"] < _first["validate_splits"]
       < _first["preflight"] < _first["build_reference"]
       < _first["assemble_gen_scores"],
       f"and `run()`'s CHEAP GATES ALL PRECEDE THE FEED, asserted from "
       f"the parse: validate_outdir(L{_first['validate_outdir']}) < "
       f"validate_splits(L{_first['validate_splits']}) < "
       f"preflight(L{_first['preflight']}) < "
       f"build_reference(L{_first['build_reference']}) < "
       f"assemble_gen_scores(L{_first['assemble_gen_scores']}). This is "
       f"what makes the `run(None, splits=None)` known-bad above cost "
       f"milliseconds instead of half an hour -- cheap BY CONSTRUCTION, "
       f"not by luck, and the check says which construction")
    _cellish = [{"slug": "s1", "side": HSP.SIDES[0], "gen": 0},
                {"slug": "s1", "side": HSP.SIDES[0], "gen": 1}]
    _ct = split_tally([(a["slug"], a["side"], a["gen"]) for a in _cellish],
                      {("s1", HSP.SIDES[0], 0): "train",
                       ("s1", HSP.SIDES[0], 1): "score"})
    ok(_ct == {"train": 1, "score": 1},
       f"and the per-cell stamp is driven on the SHAPE `run()` builds it "
       f"from -- `_treated_actions`' (slug, side, gen) dicts -> {_ct}. The "
       f"cell's label is over the generations IT ACTED ON, so it moves "
       f"with theta and is not the population's constant restated "
       f"eighteen times")

    ok(OUTDIR.exists() is False,
       f"and the DECLARED OUTDIR {OUTDIR.name} STILL DOES NOT EXIST after "
       f"this suite: round 44 is the producer half and the execution is a "
       f"separate act the USER rules")

    ok(HS.load_incumbent("btc")["_n_features"] == 60,
       "and the width the assembly is checked against is the FIT'S OWN "
       "(`linear_d_btc.json`'s normalisers, bound by the manifest), never "
       "a literal in this file")
    from collections import Counter as _C
    _verd = _C(r["verdict"] for r in _pin)
    ok(_pv.get("phase2_arms.py") == "BLOCKING"
       and _verd["IDENTICAL"] == 10 and _verd["ADDITIVE_DECLARED"] == 1
       and _verd["BLOCKING"] == 1
       and _verd["NOT_CALLED"] == 0 and sum(_verd.values()) == 12,
       f"and the closure is TRANSITIVE over first-party imports, bounded "
       f"by the manifest's twelve: {dict(_verd)}. THE COUNT MOVED THIS "
       f"ROUND, and the reason is the wiring, not the walk: "
       f"`assembly_preconditions` imports `phase2_arms` to read the "
       f"fragment and tape the fit consumed, and phase2_arms pulls the "
       f"rest of the fit stack behind it -- so all twelve pinned files are "
       f"now on the run's path (round 36 reached ONE, and earlier this "
       f"round five). NOT_CALLED is now empty, which is the honest "
       f"consequence of wiring the assembly's cheap half: nothing in the "
       f"fit's code is exempt from the comparison any more (DE36-C3)")
    ok("<module top-level>" in _fn_asts(
           (Path(__file__).resolve().parent
            / "harmful_exposure_rows.py").read_text()),
       "and the module's TOP-LEVEL BODY enters the comparison as its own "
       "entry -- a changed constant is exactly what a function-level diff "
       "cannot see (DE36-C4)")
    # DE37-C2/R1: the shas are LITERALS in this file, asserted FROM THE
    # PARSE -- a value computed at import from the file being checked is
    # not a pin, and that is exactly what round 37 shipped.
    _dsrc = [n for n in _ast.walk(_ast.parse(Path(__file__).read_text()))
             if isinstance(n, _ast.Assign)
             and any(getattr(t, "id", "") == "DECLARED_ADDITIVE"
                     for t in n.targets)]
    _shanodes = [v for d in _dsrc for dd in _ast.walk(d)
                 if isinstance(dd, _ast.Dict)
                 for k, v in zip(dd.keys, dd.values)
                 if isinstance(k, _ast.Constant)
                 and k.value in ("sha_at_fit", "sha_at_declaring_tip")]
    ok(len(_dsrc) == 1 and len(_shanodes) == 2 * len(DECLARED_ADDITIVE)
       and all(isinstance(v, _ast.Constant) for v in _shanodes)
       and sum(1 for v in _shanodes if v.value is None) == 2,
       f"DE37-C2: all {len(_shanodes)} declaration shas "
       f"({len(DECLARED_ADDITIVE)} entries x 2) are LITERAL "
       f"constants in `DECLARED_ADDITIVE` -- read from the parse, so a "
       f"future edit that recomputes one from the file it pins fails "
       f"HERE. Two are literal `None`, which is the declared fact that "
       f"those functions are ABSENT from the fit bytes, not a gap")
    _ref_fit = json.loads(
        (FITS / "fit_manifest.json").read_text())["fit_code_ref"]
    _here_dir = Path(__file__).resolve().parent
    _fitsrc, _tipsrc = {}, {}
    for _f in sorted({k[0] for k in DECLARED_ADDITIVE}):
        _fitsrc[_f] = _fn_asts(
            _git_show(_ref_fit, f"live/pm_research/{_f}") or "")
        _tipsrc[_f] = _fn_asts((_here_dir / _f).read_text())
    ok(all(DECLARED_ADDITIVE[k]["sha_at_fit"]
           == _ast_sha(_fitsrc[k[0]].get(k[1]))
           and DECLARED_ADDITIVE[k]["sha_at_declaring_tip"]
           == _ast_sha(_tipsrc[k[0]].get(k[1])) for k in DECLARED_ADDITIVE),
       f"and the {2 * len(DECLARED_ADDITIVE)} literals are TRUE of the "
       f"artifacts today: "
       f"{ {k[1]: DECLARED_ADDITIVE[k]['sha_at_declaring_tip'] for k in DECLARED_ADDITIVE} } "
       f"-- so the declaration describes the code that is actually there, "
       f"and the check above says it cannot describe itself")
    # DE39-R2 / DE40-R1: GROUPED BY `changed_at`, each group checked at
    # ITS OWN commit and parent -- and the grouping is DRIVEN ON TWO
    # GROUPS, because with one group in the real map a loop that stops
    # after the first is indistinguishable from one that does not.
    _g = declaration_groups()
    ok(_g and all(r["ok"] for r in _g),
       f"DE38 §2(iii): each declaration NAMES THE COMMIT THAT CHANGED THE "
       f"FUNCTION and the claim is CHECKED at both sides of it, grouped by "
       f"that commit: "
       f"{[(r['changed_at'], r['functions']) for r in _g]} -- at each "
       f"commit its functions carry the declared TIP shas, and at its "
       f"parent the declared FIT shas (two of them absent). Prose cannot "
       f"be pinned; the FACT the prose is about can be")
    # TWO REAL GROUPS: the three declarations at 851edaf, and `label_rows`
    # at 46ab455 -- a genuinely different commit of the same file, with
    # its own before/after shas, measured.
    _fixture = dict(DECLARED_ADDITIVE)
    _fixture[("harmful_exposure_rows.py", "label_rows")] = {
        "changed_at": "46ab455",
        "sha_at_fit": "4a4403ee715d88f7",
        "sha_at_declaring_tip": "905975dceed925f0",
        "reason": "FIXTURE ONLY -- never a declaration. It exists so the "
                  "grouping is exercised on a SECOND group (DE40-R1)"}
    _fg = declaration_groups(_fixture)
    ok(len(_fg) == 2 and all(r["ok"] for r in _fg)
       and [r["changed_at"] for r in _fg] == ["46ab455", "851edaf"]
       and _fg[0]["functions"] == ["label_rows"]
       and len(_fg[1]["functions"]) == 3,
       f"DE40-R1, DRIVEN ON TWO GROUPS: a fixture adding a real second "
       f"declaring commit yields {len(_fg)} groups -- "
       f"{[(r['changed_at'], r['functions']) for r in _fg]} -- and EACH is "
       f"checked at its OWN commit and parent. A loop that leaves after "
       f"the first group returns one row here and this goes red; at "
       f"`35452c0` it survived the whole suite, because the real map has "
       f"exactly one group")
    _bad_fx = dict(_fixture)
    _bad_fx[("harmful_exposure_rows.py", "label_rows")] = dict(
        _fixture[("harmful_exposure_rows.py", "label_rows")],
        sha_at_fit="deadbeefdeadbeef")
    _bg = declaration_groups(_bad_fx)
    ok(len(_bg) == 2 and not _bg[0]["ok"] and _bg[1]["ok"]
       and _bg[0]["mismatches"] == ["label_rows@46ab455^"],
       f"KNOWN-BAD on the SECOND group: a wrong parent sha there is caught "
       f"and NAMED ({_bg[0]['mismatches']}) while the first group stays "
       f"green -- so the check reaches past the group the real map "
       f"happens to have")
    # DE37-C2 DRIVEN, ON A SOURCE EDIT: the known-bad is an edited FUNCTION
    # BODY in a copy of the module directory, not a tampered dict -- the
    # state round 37's falsifier could not produce.
    with _tf.TemporaryDirectory() as _md:
        _here = Path(__file__).resolve().parent
        for _f in _here.glob("*.py"):
            (Path(_md) / _f.name).write_bytes(_f.read_bytes())
        _tgt = Path(_md) / "harmful_exposure_rows.py"
        _txt = _tgt.read_text()
        _decl_at = _txt.index("def select_v2_era(")
        _body_at = _txt.index("\n", _txt.index(":\n", _decl_at)) + 1
        _tgt.write_text(_txt[:_body_at] + "    _tampered_marker = 1\n"
                        + _txt[_body_at:])
        _ed = [r for r in pin_statuses(here=Path(_md))
               if r["path"] == "harmful_exposure_rows.py"][0]
        _un = [r for r in pin_statuses(here=Path(_md))
               if r["path"] == "harmful_candidate_manifest.py"][0]
    ok(_ed["verdict"] == "BLOCKING"
       and any("declaration stale" in str(x) for x in _ed["undeclared"]),
       f"KNOWN-BAD, DRIVEN ON AN EDITED FUNCTION BODY: one statement "
       f"inserted into `select_v2_era` re-opens the file to "
       f"{_ed['verdict']}, naming the stale declaration. Round 37 answered "
       f"ADDITIVE_DECLARED here and `verify_called_code()` PROCEEDED, "
       f"because the seal was recomputed from the edited file -- the three "
       f"declared functions were a permanent exemption (DE37-R1)")
    ok(_un["verdict"] == "IDENTICAL",
       f"POSITIVE CONTROL, same injected directory: an untouched pinned "
       f"file still reads {_un['verdict']}, so the BLOCKING above is the "
       f"edit and not the copy")
    ok([r["verdict"] for r in pin_statuses()
        if r["path"] == "harmful_exposure_rows.py"] == ["ADDITIVE_DECLARED"],
       "POSITIVE CONTROL: the real directory still reads "
       "ADDITIVE_DECLARED after that drive -- nothing in the repo was "
       "touched (the edit lives in a temporary copy)")
    _rcsrc = _ast.get_source_segment(
        Path(__file__).read_text(),
        [f for f in _ast.walk(_ast.parse(Path(__file__).read_text()))
         if isinstance(f, _ast.FunctionDef) and f.name == "run_cell"][0])
    _p4_guard = [nd for nd in _ast.walk(_ast.parse(_rcsrc))
                 if isinstance(nd, _ast.If)
                 and any(isinstance(x, _ast.Compare)
                         and getattr(x.left, "id", "") == "_rc_ctrl"
                         for x in _ast.walk(nd.test))]
    _budget_guard = [nd for nd in _ast.walk(_ast.parse(_rcsrc))
                     if isinstance(nd, _ast.If)
                     and any(getattr(x, "id", "") == "accepted"
                             for x in _ast.walk(nd.test))]
    ok(_p4_guard and _budget_guard,
       "both null-side guards are present in `run_cell`, asserted from the "
       "parse -- and, unlike round 37, each is also DRIVEN below (DE37-C4: "
       "the run cannot be the first place a rejection is seen to fire)")
    # ---- DE37-C1(c) and DE37-C4, DRIVEN ON THE RUN PATH ----------------
    # Two slugs in ONE (side, hour) stratum. Slug A carries TWO above
    # events; only the first ACTS, because cancelling it holds the side and
    # the second generation is suppressed. Slug B carries one below event.
    # So |above| = 2 and |actions| = 1 in that stratum -- the ordinary
    # case, and the one round 37's demand could not express.
    _A = "btc-updown-5m-1787579400"
    _B = "btc-updown-5m-1787579700"
    _href = {_A: {"BUY_UP": [_gen(1, 0.0, 20.0, [(5.0, 1.0, -20.0)]),
                             _gen(2, 21.0, 40.0, [(25.0, 1.0, -20.0)])],
                  "SELL_UP": []},
             _B: {"BUY_UP": [_gen(1, 0.0, 20.0, [(5.0, 1.0, 4.0)])],
                  "SELL_UP": []},
             # a third below-threshold generation, so the stratum is wide
             # enough that the known-bad demand below takes several
             # attempts before it happens to draw the acting generation
             "btc-updown-5m-1787580000": {
                 "BUY_UP": [_gen(1, 0.0, 20.0, [(5.0, 1.0, 4.0)])],
                 "SELL_UP": []}}
    _hsc = [{"t": 0.0, "slug": _A, "side": "BUY_UP", "gen": 1, "score": 0.9},
            {"t": 21.0, "slug": _A, "side": "BUY_UP", "gen": 2, "score": 0.8},
            {"t": 0.0, "slug": _B, "side": "BUY_UP", "gen": 1, "score": 0.1},
            {"t": 0.0, "slug": "btc-updown-5m-1787580000", "side": "BUY_UP",
             "gen": 1, "score": 0.05}]
    _harm = {"CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": _hsc}
    _hth = {"CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": 0.5}
    _base = run_cell(_href, _harm, good, thetas=_hth)
    _rc0 = _realised_by_stratum(
        _base["per_arm"]["CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm"],
        _gen_index(_href))
    _above_n = sum(1 for e in _hsc if e["score"] >= 0.5)
    ok(_rc0 == {("BUY_UP", 13): 1} and _above_n == 2,
       f"THE FIXTURE IS THE ORDINARY CASE, measured not asserted: the "
       f"stratum holds {_above_n} above-threshold events and the treated "
       f"arm realises {_rc0[('BUY_UP', 13)]} action -- the second above "
       f"event is HELD by the first cancel. Round 37 demanded the draw on "
       f"the ACTION count, so it could never name both")
    try:
        _ncell, _ncerr = run_cell(_href, _harm, good, draws=2,
                                  thetas=_hth), ""
    except (DiagRefused, MRC.ControlRefused) as _e:
        # A cell that cannot build its null must fail HERE, by name: the
        # demand is what decides whether it can, and a mutant that puts
        # round 37's demand back would otherwise end this suite in a
        # traceback rather than at a check.
        _ncell, _ncerr = None, f"{type(_e).__name__}: {str(_e)[:130]}"
    _np = (_ncell or {}).get("null_population") or {
        "n_draws_attempted": 0, "n_draws_accepted": 0,
        "n_rejected_by_reason": {}, "first_rejection": None}
    # The MESSAGE must survive the failure it reports: round 21 and round
    # 25 each ended a suite in a traceback from inside an `ok` label.
    _nrr = _np.get("n_rejected_by_reason") or {}
    _nfr = _np.get("first_rejection") or {"seed": "-"}
    ok(_ncell is not None and _np["n_draws_accepted"] == 2
       and _nrr.get("P4", 0) > 0
       and _np["first_rejection"] is not None,
       f"DRIVEN, RUN PATH: P4 REJECTS AND THE RUN REDRAWS -- "
       f"{_np['n_draws_attempted']} attempts, {_np['n_draws_accepted']} "
       f"accepted, rejections by reason "
       f"{ {k: v for k, v in _nrr.items() if v} } "
       f"(first at seed {_nfr['seed']}). A draw that "
       f"places the above values on A-gen1 and B-gen1 realises TWO "
       f"cancels against the treated arm's ONE, and the null never sees "
       f"it. Round 37 asserted this branch from the parse only"
       f"{(' -- REFUSED INSTEAD: ' + _ncerr) if _ncerr else ''}")
    ok(_ncell is not None
       and _np.get("n_distinct_accepted") == 1
       and _np.get("n_distinct_attempted", 0) >= 2
       and (_ncell or {}).get("null", "").startswith("DEGENERATE"),
       f"and THAT cell's own accepted set is a POINT MASS while its "
       f"ATTEMPTED set is not: {_np.get('n_distinct_accepted')} distinct "
       f"accepted of {_np.get('n_distinct_attempted')} distinct attempted, "
       f"so the cell reads `{(_ncell or {}).get('null')}`. The two "
       f"populations differ HERE, which is what makes the substitution "
       f"round 38 shipped (reporting the sampler's count as the null's) a "
       f"visible error rather than an invisible one")
    ok(_ncell is not None
       and all(_nrr.get(r, 1) == 0
               for r in ("PERM_NOT_OK", "P1", "P2", "P3")),
       f"and with the demand taken over ABOVE EVENTS, no draw is rejected "
       f"for a STREAM defect: {_nrr} -- P1-P3 and "
       f"`ok` hold for every draw, which is what (gamma) being BUILT (not "
       f"merely declared) looks like on this path")
    _kblog, _kbclass = [], ""
    try:
        run_cell(_href, _harm, good, draws=1, thetas=_hth,
                 _known_bad_demand=True, _draw_log=_kblog)
    except (DiagRefused, MRC.ControlRefused) as _e:
        _kbclass = type(_e).__name__
    ok(_kblog and not any(r["accepted"] for r in _kblog)
       and all("PERM_NOT_OK" in r["reasons"] for r in _kblog)
       and _kbclass in ("DiagRefused", "ControlRefused"),
       f"KNOWN-BAD, DRIVEN ON THE RUN PATH: round 37's demand restored "
       f"(the ACTION count) makes `permuted_stream` return `ok=False` in "
       f"this ordinary stratum -- and every one of the "
       f"{len(_kblog)} attempts is REJECTED under PERM_NOT_OK, none "
       f"accepted, before any replay. Round 37 bound that flag and read "
       f"it zero times, so the truncated stream was replayed and its "
       f"value entered the null. The run ends in {_kbclass}, by name")
    _sv_budget = globals()["DRAW_ATTEMPT_BUDGET"]
    _kbmsg = ""
    try:
        globals()["DRAW_ATTEMPT_BUDGET"] = 0
        run_cell(_href, _harm, good, draws=1, thetas=_hth)
    except DiagRefused as _e:
        _kbmsg = str(_e)
    finally:
        globals()["DRAW_ATTEMPT_BUDGET"] = _sv_budget
    ok("0 of 1 draws matched" in _kbmsg and "0 attempts" in _kbmsg,
       f"KNOWN-BAD, DRIVEN: a budget that permits no attempt REACHES "
       f"`null#2` and REFUSES, naming its accounting rather than "
       f"returning a smaller null: \"{_kbmsg[:110]}...\"")
    # DE38-C2: the refusal must carry the REASONS, not just P4's wording.
    _c2msg = ""
    try:
        globals()["DRAW_ATTEMPT_BUDGET"] = 1
        run_cell(_href, _harm, good, draws=3, thetas=_hth)
    except DiagRefused as _e:
        _c2msg = str(_e)
    finally:
        globals()["DRAW_ATTEMPT_BUDGET"] = _sv_budget
    _c2at = _c2msg.index("rejected:") if "rejected:" in _c2msg else 0
    ok("'P4'" in _c2msg and "rejected:" in _c2msg
       and "PERM_NOT_OK" not in _c2msg,
       f"DE38-C2, DRIVEN: `null#2` now carries `n_rejected_by_reason` -- "
       f"\"{_c2msg[_c2at:][:70]}...\" -- rather than "
       f"naming P4's reason for a total that counts every reason. On a "
       f"population where the rejections were all P4 the old wording "
       f"looked right, which is exactly when a wrong label is invisible. "
       f"Reasons with a zero count are omitted, so the refusal names what "
       f"actually happened")
    ok(run_cell(_href, _harm, good, draws=1,
                thetas=_hth)["null_population"]["n_draws_accepted"] == 1,
       f"POSITIVE CONTROL: with the budget restored to {_sv_budget} the "
       f"same cell builds its null, so the refusal above is the budget "
       f"and not the fixture")
    # DE37 item 6: the LAST substring check in these modules, replaced by
    # an assertion about the OBJECTS. "asserted at the source" is a claim
    # about text; this recomputes one accepted draw and compares numbers.
    _olog: list = []
    _ocell = run_cell(_href, _harm, good, draws=1, thetas=_hth,
                      _draw_log=_olog)
    _acc = [r for r in _olog if r["accepted"]][0]
    _opool = [{"slug": f"{sl}|{sd}|{g['gen']}", "side": sd,
               "hour": _hour_of(sl)}
              for sl, sides in sorted(_href.items())
              for sd in HSP.SIDES for g in sides[sd]]
    _oabove = [{"slug": f"{e['slug']}|{e['side']}|{e['gen']}"}
               for e in _hsc if e["score"] >= 0.5]
    _odrawn = MRC.draw(_opool, _oabove, seed=_acc["seed"])
    _octrl, _ook = permuted_stream(_hsc, _odrawn, 0.5, _gen_index(_href))
    _ores = arm_result(_href, _octrl, validate_cell(dict(good)), theta=0.5)
    ok(_ook and _acc["value"] is not None
       and _ores["cost_adjusted_value_cents"] == _acc["value"],
       f"AND THE NULL'S VALUES ARE A REPLAY'S, ASSERTED ON THE OBJECTS: "
       f"draw {_acc['seed']} is recomputed here from the pool and the "
       f"stream, replayed through `arm_result`, and its "
       f"`cost_adjusted_value_cents` is {_ores['cost_adjusted_value_cents']}"
       f" -- identical to the value the null recorded. Round 32 valued "
       f"each draw as a HARM SUM (the proxy the frozen §6 forbids), and "
       f"round 37 answered that with a substring check on this file")
    # ---- DE38-R3: ONE SOURCE for the event contract, DRIVEN ------------
    # A copy of the adapter with `gen` removed from the contract tuple is
    # loaded, used to build a stream, and handed to `run_cell`: the event
    # dict is built from that same tuple, so the key is absent from the
    # output and `null#3` refuses BY NAME. At `dfd4c00` the two were
    # separate lists and this died as a bare `KeyError` inside the adapter.
    import importlib.util as _ilu
    with _tf.TemporaryDirectory() as _sd:
        _cp = Path(_sd) / "de_score_stream_nogen.py"
        _cp.write_text((Path(__file__).resolve().parent
                        / "de_score_stream.py").read_text().replace(
            'REQUIRED_EVENT_KEYS = ("t", "slug", "side", "gen")',
            'REQUIRED_EVENT_KEYS = ("t", "slug", "side")', 1))
        _spec = _ilu.spec_from_file_location("de_score_stream_nogen", _cp)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _rows = [{"t": e["t"], "slug": e["slug"], "side": e["side"],
                  "gen": e["gen"]} for e in _fsc]
        _nogen_ev = _mod.score_events(
            _rows, head="q1_arrival_composed_lgbm", coin="btc",
            scorer=lambda r: 0.9, verified={"lgbm_haz_btc.txt": "x"})
    ok(all("gen" not in e for e in _nogen_ev),
       f"DE38-R3: the contract is ONE TUPLE -- removing `gen` from "
       f"`REQUIRED_EVENT_KEYS` removes it from the {len(_nogen_ev)} events "
       f"the adapter emits, because they are built from that tuple. At "
       f"`dfd4c00` the required-key list and the event construction were "
       f"two sources, so the key survived the removal")
    refuses(lambda: run_cell(_free, {
        "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": _nogen_ev},
        good, draws=1, thetas={
            "CONDVALUE_OVER_SKEWED_REF/q1_arrival_composed_lgbm": 0.5}),
        "and the runner then refuses BY NAME at `null#3` rather than "
        "dying as a bare `KeyError` inside the adapter -- which is what "
        "the ninth mutant did last round, and what DE38-R3 asked to close",
        needle="do not name their generation")
    _kbsrc = [n for n in _ast.walk(_ast.parse(Path(__file__).read_text()))
              if isinstance(n, _ast.Call)
              and getattr(n.func, "id", "") == "run_cell"
              and any(k.arg == "_known_bad_demand" for k in n.keywords)]
    ok(len(_kbsrc) == 1,
       f"and the known-bad demand has {len(_kbsrc)} call sites, ALL of "
       f"them in this selftest -- read from the parse. It exists so the "
       f"rejection can be driven on the run path (rule 15); a run that "
       f"passed it would be a run demanding on the wrong variable")
    with _tf.TemporaryDirectory() as _md2:
        _here2 = Path(__file__).resolve().parent
        for _f in _here2.glob("*.py"):
            (Path(_md2) / _f.name).write_bytes(_f.read_bytes())
        _t2 = Path(_md2) / "harmful_exposure_rows.py"
        _x2 = _t2.read_text()
        _at2 = _x2.index("\n", _x2.index(":\n",
                                         _x2.index("def join_fills("))) + 1
        _t2.write_text(_x2[:_at2] + "    _tampered_marker = 2\n" + _x2[_at2:])
        refuses(lambda: verify_called_code(here=Path(_md2)),
                "KNOWN-BAD, DRIVEN ON A SOURCE EDIT (rule 15): one "
                "statement inserted into `join_fills` -- a CALLED, "
                "UNDECLARED function -- refuses by name at `called#1`. "
                "Round 37's falsifier passed a synthetic status ROW, which "
                "tests the filter and not the path that produces it",
                needle="BLOCKING pin status")
    _nonblocking = [r for r in _pin if r["verdict"] != "BLOCKING"]
    ok(verify_called_code(_nonblocking) == _nonblocking
       and len(_nonblocking) == len(_pin) - 1
       and {r["verdict"] for r in _nonblocking} == {"IDENTICAL",
                                                    "ADDITIVE_DECLARED"},
       f"POSITIVE CONTROL on the same path, and it MATTERS more now that "
       f"the real set BLOCKS: the same real rows with the one BLOCKING "
       f"row dropped ({len(_nonblocking)} of {len(_pin)}) are ADMITTED "
       f"unchanged. So `called#1` is a FILTER on a verdict and not a wall "
       f"that refuses everything -- the half a refusal-only control never "
       f"proves (SEAT_PROTOCOL rule 16). The verdicts are the ones "
       f"`pin_statuses` computed, never a fabricated clean set")

    # ---- §5 (gamma): P1-P4 COMPUTED on the two streams ----------------
    _t_scores = [{"t": 0.0, "slug": _slug[0], "side": "BUY_UP", "gen": 1,
                  "score": 0.9},
                 {"t": 1.0, "slug": _slug[0], "side": "BUY_UP", "gen": 2,
                  "score": 0.8},
                 {"t": 2.0, "slug": _slug[0], "side": "BUY_UP", "gen": 3,
                  "score": 0.1}]
    _gidx3 = {(_slug[0], "BUY_UP", g): {"t0": float(g)} for g in (1, 2, 3)}
    _drawn3 = [f"{_slug[0]}|BUY_UP|3", f"{_slug[0]}|BUY_UP|1"]
    _ctrl3, _ok3 = permuted_stream(_t_scores, _drawn3, 0.5, _gidx3)
    _P = stream_predicates(_t_scores, _ctrl3, _drawn3, 0.5, _gidx3,
                           rc_treated={("BUY_UP", 13): 1},
                           rc_control={("BUY_UP", 13): 1})
    ok(_P["P1_key_multisets_equal"] and _P["P2_stratum_score_multisets_equal"]
       and _P["P3_drawn_carry_above_and_only_drawn"]
       and _P["P4_realised_action_counts_equal"],
       f"§5 (gamma), P1-P4 COMPUTED on the streams a draw produces: {_P}. "
       f"The reviewer's fixture shape -- an acting above event, a "
       f"NON-ACTING above event, and a below one, with the draw naming a "
       f"below generation -- is exactly this: one event per generation in "
       f"both arms, the per-stratum score multiset unchanged, the drawn "
       f"generations carrying the above values")
    # THE REVIEWER'S FIXTURE EXACTLY: the draw names ONE generation (the
    # below one) while the stratum holds TWO above values. Round 36's
    # construction then dropped a value in the `zip` and gave the drawn
    # generation both events.
    _drawn1 = [f"{_slug[0]}|BUY_UP|3"]
    _ctrl1, _ok1 = permuted_stream(_t_scores, _drawn1, 0.5, _gidx3)
    _P1 = stream_predicates(_t_scores, _ctrl1, _drawn1, 0.5, _gidx3)
    ok(_P1["P1_key_multisets_equal"] and _P1["P2_stratum_score_multisets_equal"]
       and _ok1 is False and _P1["P3_drawn_carry_above_and_only_drawn"]
       is False,
       f"and on the SAME fixture with an unhonourable draw -- one drawn "
       f"generation, two above values -- the new construction still keeps "
       f"P1/P2 (one event per generation, the multiset intact) and reports "
       f"the draw as UNHONOURABLE ({_ok1}) so it is redrawn, rather than "
       f"honouring it by dropping a value")
    _old = [dict(e) for e in _t_scores if float(e["score"]) < 0.5]
    _vals_old = [float(e["score"]) for e in _t_scores
                 if float(e["score"]) >= 0.5]
    for (k, v) in zip(sorted(_drawn1), _vals_old):
        _sl, _sd, _gn = k.split("|")
        _old.append({"t": float(_gn), "slug": _sl, "side": _sd,
                     "gen": int(_gn), "score": v})
    _Pold = stream_predicates(_t_scores, _old, _drawn1, 0.5, _gidx3)
    ok(not _Pold["P1_key_multisets_equal"]
       and not _Pold["P2_stratum_score_multisets_equal"],
       f"KNOWN-BAD: round 36's construction on that fixture reads "
       f"{ {k: v for k, v in _Pold.items() if v is not None} } -- **P1 and "
       f"P2 RED**. It built the stream from the drawn keys alone, so the "
       f"drawn generation carried two events and the others none, and the "
       f"`zip` dropped an above value: measured [0.1, 0.9] against the "
       f"treated [0.1, 0.8, 0.9] (DE36-C1)")
    _P4bad = stream_predicates(_t_scores, _ctrl3, _drawn3, 0.5, _gidx3,
                               rc_treated={("BUY_UP", 13): 1},
                               rc_control={("BUY_UP", 13): 2})
    ok(_P4bad["P4_realised_action_counts_equal"] is False,
       "and P4 is the DECISION variable, separately: a permutation that "
       "changes the per-stratum realised action count is REJECTED and "
       "redrawn (counted in `n_rejected_by_stratum`), because a stateful "
       "policy cannot be made to cancel exactly the drawn set -- which is "
       "why `control#2` is withdrawn (DE36-R3)")
    # ---- DE37-C3: the below values stay at their own generations ------
    _bs = [{"t": 0.0, "slug": _slug[0], "side": "BUY_UP", "gen": 1,
            "score": 0.9},
           {"t": 1.0, "slug": _slug[0], "side": "BUY_UP", "gen": 2,
            "score": 0.2},
           {"t": 2.0, "slug": _slug[0], "side": "BUY_UP", "gen": 3,
            "score": 0.1}]
    _gidxb = {(_slug[0], "BUY_UP", g): {"t0": float(g)} for g in (1, 2, 3)}
    _bdrawn = [f"{_slug[0]}|BUY_UP|3"]
    _bctrl, _bok = permuted_stream(_bs, _bdrawn, 0.5, _gidxb)
    _bmap = {(e["slug"], e["side"], e["gen"]): e["score"] for e in _bctrl}
    _k = lambda g: (_slug[0], "BUY_UP", g)
    ok(_bok and _bmap[_k(3)] == 0.9 and _bmap[_k(2)] == 0.2
       and _bmap[_k(1)] == 0.1,
       f"DE37-C3: THE BELOW VALUES STAY AT THEIR OWN GENERATIONS. The "
       f"draw names gen 3, so it takes the above value 0.9 and gen 1 takes "
       f"the below value gen 3 gave up (0.1) -- and gen 2, drawn by "
       f"nobody, KEEPS ITS OWN 0.2. Round 37 sorted the below values "
       f"descending onto the non-drawn keys in stream order, which put "
       f"0.2 on gen 1 and 0.1 on gen 2: both moved, for no reason (γ) "
       f"asks for")
    _sorted_would = dict(zip([_k(1), _k(2)], sorted([0.1, 0.2],
                                                    reverse=True)))
    ok(_sorted_would[_k(2)] != _bmap[_k(2)],
       f"KNOWN-BAD, COMPUTED: round 37's descending assignment would put "
       f"{_sorted_would[_k(2)]} on gen 2 where its own value is "
       f"{_bmap[_k(2)]}. Repost eligibility is a DWELL condition on the "
       f"below path, so a moved below value changes when a held side "
       f"becomes repost-eligible -- §2's number meeting §5's stream")
    # ---- DE37-R3: P3 asks whether the draw is IN the stream ------------
    # The vacuous shape needs a stream with NO above events: then the
    # control carries no above key, the filtered draw is empty too, and
    # round 37's P3 compared {} == {} and answered True.
    _allbelow = [dict(x, score=0.1 + 0.01 * i)
                 for i, x in enumerate(_bs)]
    _acctrl, _acok = permuted_stream(_allbelow, [], 0.5, _gidxb)
    _Pvac = stream_predicates(_allbelow, _acctrl,
                              [f"{_slug[0]}|BUY_UP|9"], 0.5, _gidxb)
    _Pvac_old = ({(e["slug"], e["side"], e["gen"]) for e in _acctrl
                  if e["score"] >= 0.5}
                 == {w for w in {(_slug[0], "BUY_UP", 9)}
                     if w in {(e["slug"], e["side"], e["gen"])
                              for e in _acctrl}})
    _Pempty = stream_predicates(_bs, _bctrl, [], 0.5, _gidxb)
    ok(_Pvac["P3_drawn_carry_above_and_only_drawn"] is False
       and _Pvac_old is True
       and _Pempty["P3_drawn_carry_above_and_only_drawn"] is False,
       f"DE37-R3, THE VACUOUS SHAPE DRIVEN AND ITS OLD ANSWER COMPUTED "
       f"BESIDE IT: on an all-below stream with the draw naming gen 9 -- a "
       f"generation the stream does not carry -- round 37's P3 compared "
       f"two empty sets and answered {_Pvac_old}; asking `want <= "
       f"keys(stream)` FIRST answers "
       f"{_Pvac['P3_drawn_carry_above_and_only_drawn']}. An EMPTY draw is "
       f"red for the same reason: a null draw that names nothing is not a "
       f"draw whose keys all carry above values")
    ok(len(_drawn3) == 2 and sum(1 for e in _t_scores
                                 if e["score"] >= 0.5) == 2,
       f"DE37-R2 CLOSED BY THE SAME CHANGE: the (gamma) fixture above "
       f"draws {len(_drawn3)} keys where the stratum holds 2 above events "
       f"-- which round 37's demand (the ACTION count) could not produce, "
       f"so P1-P3 were asserted only on a state the run path could not "
       f"reach. With the demand over above events, that IS the run path's "
       f"state, and the driven checks earlier in this suite exercise it "
       f"through `run_cell`")
    _np2 = ncell.get("null_population") or {}
    ok(_np2.get("n_draws_accepted") == _np2.get("n_draws_attempted", -1)
       - sum([_np2.get("n_draws_attempted", 0)
              - _np2.get("n_draws_accepted", 0)])
       or _np2.get("n_draws_accepted", 0) >= 1,
       f"and the receipt carries the attempt accounting: attempted "
       f"{_np2.get('n_draws_attempted')}, accepted "
       f"{_np2.get('n_draws_accepted')}, rejected by stratum "
       f"{_np2.get('n_rejected_by_stratum')}, budget "
       f"{_np2.get('draw_attempt_budget')}")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_phase4_diag_runner] selftest OK -- {n[0]} checks")
    return 0


def run(outdir: Path | None = None, *, splits, coins=COINS,
        limit: int | None = None) -> dict:
    """THE RUN PATH (DE32-C1).  Feed -> assembly -> scores -> arms -> rho ->
    null -> receipt, written once into the declared directory.

    Round 32 declared an invocation whose flag did not exist: `main()`
    parsed `--selftest` only, so `--run` exited argparse rc 2 while the
    filing named it as the invocation of record. The flag is real here and
    the path under it is the one the addendum declares.

    `splits` is a REQUIRED keyword with NO default. R-496 (E) ruled the
    set; the ruling is recorded in `SPLIT_RULING` and is still not a
    fallback here, so a caller who names nothing gets a refusal by name
    rather than a run under a value nobody typed."""
    out = validate_outdir(outdir or OUTDIR)
    got = validate_splits(splits)
    preflight(splits=got)      # DE34-C1: BEFORE the feed, not after it
    t_feed = time.time()
    feeds = {c: build_reference(c, limit=limit) for c in coins}
    feed_s = time.time() - t_feed
    # THE EXPENSIVE HALF, once. It is the same tape index and the same
    # feature pass for every cell, and round 43's wiring put it behind a
    # per-(coin, head) call -- 18 cells x 2 heads would have paid for it
    # 36 times.
    t_asm = time.time()
    asm = assemble_gen_scores({c: feeds[c]["reference"] for c in coins},
                              splits=got, coins=coins)
    asm_s = time.time() - t_asm
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
                    ev = score_events_for(
                        ref, coin=coin, head=head,
                        gen_scores=asm["by_arm"][(coin, head)][0])
                    scores[arm] = ev
                    thetas[arm] = theta_for(coin, head, budget)
                draws = (N_DRAWS
                         if (coin, L, budget) in
                         {(c, l, b) for c, l, b in NULL_CELLS} else 0)
                cell_out = run_cell(ref, scores, cell, draws=draws,
                                    thetas=thetas)
                # R-496 (E): SPLITS LABELLED PER CELL -- and the label is
                # over the generations THIS cell acted on, which is a
                # per-cell fact (theta moves with the budget), not the
                # population's constant restated eighteen times.
                _acts = [(a["slug"], a["side"], a["gen"])
                         for arm in cell_out["per_arm"].values()
                         for a in _treated_actions(arm)]
                cell_out["splits"] = {
                    "declared_set": asm["assembly"]["split_set"],
                    "splits": list(asm["assembly"]["splits"]),
                    "ruled_by": SPLIT_RULING["ruled_by"],
                    "population_generations_by_split":
                        asm["split_counts"][coin],
                    "treated_actions_by_split": split_tally(
                        _acts, asm["split_by_gen"][coin]),
                    "n_treated_actions": len(_acts),
                }
                cell_out["feed"] = {
                    # Published so the FIRST REAL RUN prices the replay
                    # instead of projecting it from a one-generation
                    # fixture (item 5 / DE36-C6).
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
    rec["assembly_seconds"] = asm_s
    rec["assembly"] = asm["assembly"]
    rec["split_ruling"] = dict(SPLIT_RULING)
    validate_receipt(rec)
    out.mkdir(parents=True, exist_ok=False)
    (out / "phase4_diag_r459_receipt.json").write_text(
        json.dumps(rec, indent=1, sort_keys=True))
    return rec


def score_events_for(reference: dict, *, coin: str, head: str,
                     gen_scores: dict | None = None) -> list:
    """Score events for every generation in the reference, through the
    manifest-bound adapter -- never a stub."""
    v = SS.verify_head(head, coin)
    rows = [{"t": g["t0"], "slug": slug, "side": side, "gen": g["gen"]}
            for slug, sides in sorted(reference.items())
            for side in HSP.SIDES for g in sides[side]]
    return SS.score_events(rows, head=head, coin=coin,
                           scorer=_head_scorer(head, coin, gen_scores),
                           verified=v)


def assembly_preconditions() -> dict:
    """Everything about the feature assembly that is knowable in
    milliseconds, MEASURED off the artifacts the fit named.

    The assembly's expensive half (a tape index over 3.2 GB and a feature
    pass over 1.1M rows) cannot tell you it is misconfigured until it has
    run. Each fact below decides whether it is worth starting, and each is
    read from `fit_manifest.json` or the file itself -- never from a
    literal in this file."""
    import phase2_arms as PA
    man = json.loads((FITS / "fit_manifest.json").read_text())
    out: dict = {"fragment_path": str(PA.FRAGMENT),
                 "tape_path": str(PA.TAPE_PATH)}
    for what, live, declared_p, declared_b in (
            ("fragment", PA.FRAGMENT, man.get("fragment_path"),
             man.get("fragment_bytes")),
            ("tape", PA.TAPE_PATH, man.get("tape_path"),
             man.get("tape_bytes"))):
        if not Path(live).exists():
            # SITE: assembly#2
            raise DiagRefused(
                f"the {what} the fit consumed ({live}) is absent, so the "
                f"feature pass would build features from a DIFFERENT input "
                f"than the heads were fitted on")
        if str(live) != str(declared_p):
            # SITE: assembly#3
            raise DiagRefused(
                f"phase2_arms points at {live} for the {what} and the fit "
                f"manifest names {declared_p}. Two paths is two "
                f"populations; the manifest is the fit's own record")
        nb = Path(live).stat().st_size
        if declared_b is not None and nb != declared_b:
            # SITE: assembly#4
            raise DiagRefused(
                f"the {what} is {nb:,} bytes and the fit recorded "
                f"{declared_b:,}: the file MOVED after the fit, so the "
                f"features it yields are not the fitted ones (the sha is "
                f"the fit's own check; this is the cheap one that runs "
                f"before a 3.2 GB read)")
        out[f"{what}_bytes"] = nb
    out["incumbent_width"] = HS.load_incumbent(COINS[0])["_n_features"]
    out["lgbm_width"] = HS.load_lgbm(COINS[0])[1]
    out["lgbm_norm_width"] = HS.load_lgbm_normalisers(COINS[0])["n_raw"]
    out["state_width"] = out["lgbm_norm_width"] - out["incumbent_width"]
    return out


#: The two split sets round 43 put to the USER, and the one the USER
#: RULED. Addendum v2 ask 1a, adopted at R-496 (E) as recommended:
#: **MECHANICS ON BOTH SPLITS, with splits labelled per cell.**
DECLARED_SPLIT_SETS = {
    "MECHANICS_BOTH_SPLITS": ("score", "train"),
    "SCORE_ONLY": ("score",),
}
TAPE_SPLITS = ("score", "train")

#: THE RULING, recorded -- and deliberately NOT a default anywhere in this
#: file. `run(splits=...)` is a REQUIRED keyword and `validate_splits`
#: refuses silence by name, so a caller that never chose still cannot
#: proceed under a value it did not name. The distance between "the USER
#: ruled X" and "the code assumes X when nobody says anything" is the
#: whole of rule 14: this constant is what an operator NAMES on the
#: command line, not what the runner falls back to.
RULED_SPLIT_SET = "MECHANICS_BOTH_SPLITS"
SPLIT_RULING = {
    "set": RULED_SPLIT_SET,
    "ruled_by": "R-496 (E)",
    "ask": "addendum v2 §1a",
    "adopted": "2026-09-03T03:57Z",
    "labelling": "splits labelled per cell",
    "consequence_stated_in_the_ruling": (
        "the §3 population spans BOTH fit splits (1,125,289 train rows and "
        "638,917 score rows), so cells score generations the heads were "
        "FITTED on. That is what MECHANICS means: the run measures whether "
        "the machinery composes and replays, and it is never evidence "
        "about the heads' skill"),
}


def tape_rows_array_closed(path: Path | None = None, *,
                           tail_bytes: int = 64) -> dict:
    """Does the tape's rows array actually END with `]`?

    The factual anchor of the `_stream_tape_rows` declaration. The added
    refusal fires only at EOF WITHOUT the closing bracket, so whether it
    can fire on this input is a question about the input's last bytes --
    which is checkable in microseconds and is checked, rather than
    asserted in the declaration's prose."""
    import phase2_arms as PA
    p = Path(path) if path is not None else PA.TAPE_PATH
    if not p.exists():
        # SITE: tape#1
        raise DiagRefused(f"{p} does not exist, so nothing can be said "
                          f"about how it ends")
    with open(p, "rb") as fh:
        fh.seek(0, 2)
        n = fh.tell()
        fh.seek(max(0, n - tail_bytes))
        tail = fh.read()
    txt = tail.decode("utf-8", "replace").rstrip()
    closed = txt.endswith("]") or (txt.endswith("}")
                                   and "]" in txt[-8:])
    return {"path": str(p), "bytes": n, "tail": txt[-24:],
            "rows_array_closed": bool(closed)}


def _fn_node(src: str, name: str):
    """The top-level FunctionDef `name` in `src`, or None."""
    import ast as _a
    for nd in _a.parse(src).body:
        if isinstance(nd, (_a.FunctionDef, _a.AsyncFunctionDef)) \
                and nd.name == name:
            return nd
    return None


def stream_tape_rows_drift(*, candidate: str | None = None,
                           tip_src: str | None = None) -> dict:
    """THE UNDECLARED DRIFT, COMPUTED -- every clause a predicate.

    Round 43 wrote four sentences about `_stream_tape_rows` and they were
    all true; sentences are still not evidence (rule 10). This function
    computes each of them:

    * `accepting_path_unchanged` by SUBSTITUTION, which is the only form
      of the claim that is checkable. Put a bare `return` back where the
      tip raises, and if the whole function's AST then equals the fit
      commit's, the difference IS that one statement and nothing else --
      every other path, including the accepting one, is identical. A
      textual diff cannot say that; this can.
    * `enclosing_test`, so the branch the change lives in is named from
      the parse rather than from a reading of the comment beside it.
    * `changed_at_verified`, by checking the candidate commit from BOTH
      sides -- the tip sha at it, the fit sha at its parent.
    * `rows_array_closed`, from the tape's own last bytes.

    `candidate` and `tip_src` are injectable for the falsifiers ONLY: a
    wrong commit must read False here, and a tip whose difference is NOT
    one statement must read `accepting_path_unchanged` False. The run
    passes neither."""
    import ast as _a
    name = UNDECLARED_DRIFT["function"]
    fname = UNDECLARED_DRIFT["file"]
    ref = json.loads((FITS / "fit_manifest.json").read_text())["fit_code_ref"]
    fit_src = _git_show(ref, f"live/pm_research/{fname}")
    if fit_src is None:
        # SITE: drift#1
        raise DiagRefused(
            f"the fit bytes of {fname} are not retrievable at {ref}: with "
            f"no left-hand side there is no drift to describe, and a "
            f"report that quietly compares the tip with itself is worse "
            f"than none")
    if tip_src is None:
        tip_src = (Path(__file__).resolve().parent / fname).read_text()
    A, B = _fn_node(fit_src, name), _fn_node(tip_src, name)
    if A is None or B is None:
        # SITE: drift#2
        raise DiagRefused(
            f"{name} is absent from {fname} on "
            f"{'the fit side' if A is None else 'the tip side'}: an absent "
            f"function is a different fact from a changed one and is not "
            f"reported as one")
    want = _a.dump(A, annotate_fields=True, include_attributes=False)
    raises = [nd for nd in _a.walk(B) if isinstance(nd, _a.Raise)]
    hits = []
    for nd in raises:
        keep = nd.exc
        nd.exc, nd.cause = None, None
        nd.__class__ = _a.Return
        nd.value = None
        got = _a.dump(B, annotate_fields=True, include_attributes=False)
        nd.__class__ = _a.Raise
        nd.exc, nd.cause, nd.value = keep, None, None
        if got == want:
            hits.append(nd)
    node = hits[0] if len(hits) == 1 else None
    encl = None
    if node is not None:
        for nd in _a.walk(B):
            if isinstance(nd, _a.If) and any(
                    st is node for st in nd.body):
                encl = _a.unparse(nd.test)
                break
    at = _fn_asts(_git_show(candidate or UNDECLARED_DRIFT[
        "candidate_changed_at"], f"live/pm_research/{fname}") or "")
    before = _fn_asts(_git_show(
        (candidate or UNDECLARED_DRIFT["candidate_changed_at"]) + "^",
        f"live/pm_research/{fname}") or "")
    sha_fit = _ast_sha(_fn_asts(fit_src).get(name))
    sha_tip = _ast_sha(_fn_asts(tip_src).get(name))
    return {
        "file": fname, "function": name, "fit_code_ref": ref,
        "sha_at_fit": sha_fit, "sha_at_tip": sha_tip,
        "differs": sha_fit != sha_tip,
        "candidate_changed_at": candidate or UNDECLARED_DRIFT[
            "candidate_changed_at"],
        "changed_at_verified": (at.get(name) is not None
                                and _ast_sha(at.get(name)) == sha_tip
                                and _ast_sha(before.get(name)) == sha_fit),
        "n_substitutions_that_restore_the_fit": len(hits),
        "accepting_path_unchanged": node is not None,
        "changed_statement": ("Return -> Raise" if node is not None
                              else None),
        "enclosing_test": encl,
        "raise_message_head": (
            _a.unparse(node.exc)[:80] if node is not None else None),
        "tape": tape_rows_array_closed(),
        "how_accepting_path_unchanged_was_established": (
            "SUBSTITUTION: replacing the single Raise with a bare Return "
            "makes the tip function's AST equal the fit commit's, so the "
            "difference is that one statement and no other path moved"),
    }


def code_drift_report() -> dict:
    """The pin's verdict and the undeclared drift, COMPUTED and reportable
    without running anything -- because the decision the drift needs is the
    USER's and a decision needs its facts in front of it.

    `blocking` here is derived from `pin_statuses()`, never from a literal:
    if the USER grants the declaration, this function's verdict changes on
    its own."""
    rows = pin_statuses()
    blocking = {r["path"]: r["functions_changed"] for r in rows
                if r["verdict"] == "BLOCKING"}
    drift = stream_tape_rows_drift()
    return {
        "pin": rows,
        "verdicts": {r["path"]: r["verdict"] for r in rows},
        "blocking": blocking,
        "run_is_blocked_by_the_pin": bool(blocking),
        "undeclared_drift": dict(UNDECLARED_DRIFT, **{"computed": drift}),
        "decides": "nothing -- admissibility is the USER's (rule 14); "
                   "these are the facts the ruling needs",
    }


def validate_splits(splits) -> tuple:
    """The split set, as an ARGUMENT. Never a default (rule 14).

    The §3 population spans both of the fit's splits, so scoring it on
    both means scoring generations the heads were FITTED on. That is a
    decision with a priced trade-off -- a mechanics diagnostic on a
    consumed population, or a smaller population that was never fitted --
    and it belongs to the USER. This function's whole job is to refuse
    the state where nobody chose."""
    if splits is None:
        # SITE: splits#1
        raise DiagRefused(
            f"the split set is UNDECLARED. The §3 population spans BOTH "
            f"fit splits, so every cell would score generations the heads "
            f"were fitted on -- or not -- and which of those a cell means "
            f"is not a default this runner may pick. The USER HAS RULED "
            f"({SPLIT_RULING['ruled_by']}: {RULED_SPLIT_SET}) and the "
            f"ruling is still not a default here: NAME it "
            f"({sorted(DECLARED_SPLIT_SETS)}, or an explicit subset of "
            f"{list(TAPE_SPLITS)}) and it runs; leave it silent and it "
            f"refuses HERE, before any feed. A ruling that the code "
            f"supplies on the caller's behalf is a ruling nobody has to "
            f"read (rule 14)")
    got = tuple(sorted({str(x) for x in splits}))
    if not got:
        # SITE: splits#2
        raise DiagRefused("an EMPTY split set selects no rows at all: the "
                          "feature pass would drop every generation and "
                          "the run would read as a null result")
    bad = [x for x in got if x not in TAPE_SPLITS]
    if bad:
        # SITE: splits#3
        raise DiagRefused(
            f"split(s) {bad} are not in the tape ({list(TAPE_SPLITS)}): a "
            f"name the tape does not carry indexes nothing, and an empty "
            f"index is indistinguishable from a population that was "
            f"excluded")
    return got


def split_set_name(splits) -> str:
    """The declared name of a split set, or an explicit description."""
    for name, val in DECLARED_SPLIT_SETS.items():
        if tuple(sorted(val)) == tuple(sorted(splits)):
            return name
    return "EXPLICIT:" + "+".join(sorted(splits))


def feature_blocks(*, splits, fragment: Path | None = None) -> dict:
    """THE EXPENSIVE HALF, wired: the fit's own tape index over the
    DECLARED splits, and the fit's own feature pass over the fragment.

    Nothing here is re-implemented. `PA.tape_index` and
    `PA._feature_pass` are the functions the fit ran; this supplies the
    split set (an argument, never a default) and reports what each stage
    cost and what it excluded.

    Returns the per-coin blocks, the per-key split map (so a cell can say
    which splits it consumed), the timings, and the pass's own drop
    counts -- exclusions are STATUSES here, never silent (rule 4)."""
    got = validate_splits(splits)
    pre = assembly_preconditions()
    import phase2_arms as PA
    frag = Path(fragment) if fragment is not None else PA.FRAGMENT
    stages: dict = {}
    TAPE: dict = {}
    split_of: dict = {}
    for sp in got:
        t0 = time.time()
        idx = PA.tape_index(sp)
        dup = [k for k in idx if k in TAPE]
        if dup:
            # SITE: assembly#5
            raise DiagRefused(
                f"{len(dup)} tape key(s) appear in more than one split "
                f"(first {dup[0]}): the splits are supposed to partition "
                f"the tape, and a key in both would be indexed twice and "
                f"scored under whichever split happened to be read last")
        TAPE.update(idx)
        for k in idx:
            split_of[k] = sp
        stages[f"tape_index[{sp}]"] = {
            "wall_s": round(time.time() - t0, 2),
            "rows_indexed": len(idx),
            "peak_rss_mb_highwater": _peak_rss_mb()}
    t1 = time.time()
    blocks = PA._feature_pass(frag, "phase4_diag", TAPE=TAPE)
    stages["_feature_pass"] = {
        "wall_s": round(time.time() - t1, 2),
        "fragment": str(frag),
        "kept_by_coin": {c: len(b["kept"]) for c, b in blocks.items()},
        "drops_by_coin": {c: dict(b["drops"]) for c, b in blocks.items()},
        "peak_rss_mb_highwater": _peak_rss_mb()}
    # The width the fit itself carries, read from the artifacts the
    # manifest binds -- never a literal in this file (rule 15's positive
    # control has to be able to fail).
    for coin, b in blocks.items():
        if not b["kept"]:
            continue
        w = len(b["PM"][0]) + len(b["FN"][0])
        if w != pre["incumbent_width"]:
            # SITE: assembly#6
            raise DiagRefused(
                f"the assembled PM+fine vector for {coin} is {w} wide and "
                f"the fit's own incumbent normalises "
                f"{pre['incumbent_width']}: the pass produced vectors no "
                f"head was fitted on, which would score as numbers and "
                f"not as predictions")
        if len(b["ST"][0]) + w != pre["lgbm_norm_width"]:
            # SITE: assembly#7
            raise DiagRefused(
                f"{coin}: PM+fine+state is {len(b['ST'][0]) + w} against "
                f"the fit's {pre['lgbm_norm_width']} normalisers")
    return {"blocks": blocks, "splits": got,
            "split_set": split_set_name(got), "split_of": split_of,
            "stages": stages, "widths": pre,
            "n_tape_rows": len(TAPE)}


def scratch_outdir(path) -> Path:
    """A MEASUREMENT directory: never under `data/`, never the declared
    OUTDIR.

    The cost measurement writes slices of the fit's own inputs, and a
    slice of the tape landing beside the tape is how a diagnostic input
    becomes a production one. Both guards are identity checks on resolved
    paths, so a symlink into `data/` cannot walk around them."""
    p = Path(path).resolve()
    if p == OUTDIR.resolve() or OUTDIR.resolve() in p.parents:
        # SITE: measure#1
        raise DiagRefused(
            f"{p} is (or is under) the DECLARED diagnostic OUTDIR "
            f"{OUTDIR}: that directory is created by the RUN, once, and a "
            f"measurement that pre-creates it turns the run's "
            f"already-exists refusal into a false alarm")
    data = (ROOT / "data").resolve()
    if p == data or data in p.parents:
        # SITE: measure#2
        raise DiagRefused(
            f"{p} is under {data}: this seat is READ-ONLY under `data/` "
            f"and a sliced copy of the tape or the fragment written there "
            f"is an artifact nobody declared, sitting beside the ones the "
            f"fit is bound to")
    p.mkdir(parents=True, exist_ok=True)
    return p


def input_roots(*, archive_repo: str | None = None) -> dict:
    """WHERE THE FIT STACK READS FROM -- and it is TWO roots, not one.

    `phase2_arms.DERIVED` is a HARDCODED absolute path: the tree the fit
    ran in. `flow_intensity.PM` -- which is where the WINDOW ARCHIVES and
    `markets.jsonl` come from -- is `__file__`-relative. In the main tree
    the two coincide and nothing is visible. From ANY worktree they do
    not, and `_feature_pass` then reads the tape and the fragment from one
    tree and the archives from another.

    This is not hypothetical and it is not a code defect anyone can fix
    from this seat: measured here, the disagreement makes `_archive_paths`
    return nothing for all 471 §3 windows, and the fit's own absorption
    bound refuses at `no_archive` 100%. That is the guard working. What it
    must never be is a surprise at minute 29 of a run, so it is computed
    BEFORE anything expensive."""
    import phase2_arms as PA
    import flow_intensity as fi
    d = Path(PA.DERIVED).resolve().parents[2]
    # `archive_repo` is injectable ONLY so both directions of the
    # comparison can be driven -- a guard that has only ever seen
    # DISAGREE has not been shown to recognise AGREE (rule 16).
    a = (Path(archive_repo).resolve() if archive_repo is not None
         else Path(fi.PM).resolve().parents[1])
    return {"derived_root": str(d), "archive_root": str(a),
            "agree": d == a,
            "derived": str(Path(PA.DERIVED).resolve()),
            "archive_raw": str(Path(fi.RAW).resolve()),
            "archive_raw_exists": Path(fi.RAW).is_dir()}


def _write_rows(dst: Path, rows: list) -> dict:
    """Write a `{"rows": [...]}` file the fit's own readers accept."""
    dst.write_text(json.dumps({"rows": rows}))
    return {"path": str(dst), "bytes": dst.stat().st_size,
            "n_rows": len(rows)}


def fragment_slice(dst: Path, *, n_windows: int, source: Path | None = None,
                   only_slugs=None, row_cap: int = 400_000) -> dict:
    """The first `n_windows` WINDOWS of the fragment, whole.

    Windows, not rows: `_feature_pass` groups by slug and reads one
    archive per slug, so a slice cut mid-window measures a pass whose
    per-window fixed cost is charged to a fraction of its rows.

    `only_slugs` restricts the slice to windows the pass can actually DO
    WORK ON -- the §3 population that has an archive and a token pair. A
    slice of windows the pass drops at the first branch measures the
    drop, and the extrapolation from it would be an order of magnitude
    too cheap."""
    import phase2_arms as PA
    src = Path(source) if source is not None else PA.FRAGMENT
    want = set(only_slugs) if only_slugs is not None else None
    keep: list = []
    slugs: list = []
    scanned = 0
    for r in PA._stream_tape_rows(src):
        scanned += 1
        sl = r["slug"]
        if want is not None and sl not in want:
            if len(slugs) >= n_windows:
                break
            continue
        if sl not in slugs:
            if len(slugs) >= n_windows:
                break
            slugs.append(sl)
        keep.append(r)
        if len(keep) >= row_cap:
            # SITE: measure#3
            raise DiagRefused(
                f"the first {n_windows} window(s) of {src.name} exceed the "
                f"{row_cap:,}-row cap: the slice is meant to be small, and "
                f"silently returning a truncated window would measure a "
                f"pass over rows whose archive cost was paid once")
    out = _write_rows(Path(dst), keep)
    out["slugs"] = slugs
    out["n_windows"] = len(slugs)
    out["fragment_rows_scanned"] = scanned
    out["coins"] = sorted({r["coin"] for r in keep})
    if not slugs:
        # SITE: measure#9
        raise DiagRefused(
            f"the slice is EMPTY: no window of {src.name} matched the "
            f"{0 if want is None else len(want)} wanted slugs. An empty "
            f"slice measures nothing and would extrapolate from zero")
    return out


def tape_slice(dst: Path, *, slugs, source: Path | None = None,
               row_cap: int = 2_000_000) -> dict:
    """Every tape row belonging to `slugs`, so the join is COMPLETE.

    A slice that half-covers its fragment measures `state_join_failed`,
    not the pass -- and `assert_fit_absorption_within_bound` would refuse
    it, which is the fit's own guard telling you the measurement is about
    the slice."""
    import phase2_arms as PA
    src = Path(source) if source is not None else PA.TAPE_PATH
    want = set(slugs)
    keep: list = []
    seen: set = set()
    n = 0
    t0 = time.time()
    for r in PA._stream_tape_rows(src):
        n += 1
        sl = r["slug"]
        if sl in want:
            keep.append(r)
            seen.add(sl)
        elif seen and len(seen) == len(want):
            break                 # past the wanted block: tape is grouped
        if n >= row_cap:
            # SITE: measure#4
            raise DiagRefused(
                f"scanned {n:,} tape rows without closing the slice "
                f"({sorted(want - seen)} still unseen): the slice would "
                f"join partially and the pass would measure join failures")
    out = _write_rows(Path(dst), keep)
    out.update({"slugs_covered": sorted(seen),
                "tape_rows_scanned": n,
                "scan_wall_s": round(time.time() - t0, 2),
                "splits_present": sorted({str(r.get("split")) for r in keep})})
    return out


def row_density(path: Path, *, sample_bytes: int = 32 << 20) -> dict:
    """Bytes per row and the implied row count, read from the ORIGINAL file.

    The obvious basis for an extrapolation -- the slice this runner wrote
    -- is BIASED: `json.dumps` re-serialises with its own separators, so a
    slice is a few percent larger per row than the bytes it came from, and
    every count derived from it comes out low. The density is therefore
    measured on the artifact itself, from a sample whose consumed offset
    is tracked exactly."""
    p = Path(path)
    n = p.stat().st_size
    dec = json.JSONDecoder()
    with open(p, "r") as fh:
        head = fh.read(1 << 16)
        i = head.index('"rows"')
        i = head.index("[", i) + 1
        header = i
        buf = head[i:] + fh.read(max(0, sample_bytes - len(head)))
    # INDEXED, never sliced: `buf[used:]` on a 32 MB sample is quadratic
    # and turned a two-second density read into a two-minute timeout.
    used, rows, n_buf = 0, 0, len(buf)
    while used < n_buf:
        while used < n_buf and buf[used] in " \t\r\n,":
            used += 1
        if used >= n_buf or buf[used] == "]":
            break
        try:
            _, end = dec.raw_decode(buf, used)
        except ValueError:
            break
        used = end
        rows += 1
    if not rows:
        # SITE: measure#11
        raise DiagRefused(
            f"no complete row decoded from the first {sample_bytes:,} "
            f"bytes of {p.name}: a density of zero rows per byte would "
            f"make every extrapolation infinite")
    bpr = used / rows
    return {"path": str(p), "file_bytes": n, "header_bytes": header,
            "sample_bytes_consumed": used, "rows_in_sample": rows,
            "bytes_per_row": bpr,
            "estimated_total_rows": (n - header) / bpr,
            "basis": "the ORIGINAL file, not a re-serialised slice"}


#: Linearity is an ASSUMPTION, and it is written down beside every number
#: it produces rather than left in a reader's head.
LINEARITY_ASSUMPTION = (
    "EXTRAPOLATION, LINEAR IN THE STATED UNIT. The measured slice is "
    "scaled by (full unit / slice unit). This assumes cost per unit is "
    "CONSTANT across the artifact -- it is not measured to be, and three "
    "things could break it: per-window archive costs vary with window "
    "size, the tape's split mix varies along the file, and `_feature_pass` "
    "materialises the WHOLE fragment with `json.loads(read_text())`, so "
    "its MEMORY is a property of the full file and does not scale with a "
    "row slice at all. Treat the wall-clock figures as an order of "
    "magnitude and the memory figure as a lower bound.")


def extrapolate(measured: float, *, slice_unit: float, full_unit: float,
                unit: str) -> dict:
    """A scaled number that carries its own provenance (rule 10)."""
    if slice_unit <= 0:
        # SITE: measure#5
        raise DiagRefused(
            f"an extrapolation from a slice of {slice_unit} {unit} is a "
            f"division by nothing: a number with no measurement under it "
            f"is not an estimate")
    f = full_unit / slice_unit
    return {"provenance": "EXTRAPOLATION",
            "measured": measured, "unit": unit,
            "slice_unit": slice_unit, "full_unit": full_unit,
            "factor": f, "estimate": measured * f,
            "assumption": LINEARITY_ASSUMPTION}


def feature_pass_materialises_whole_file() -> dict:
    """Is `_feature_pass`'s input read whole? COMPUTED from its parse.

    The memory claim in `LINEARITY_ASSUMPTION` is the kind of sentence
    this programme has been wrong about; it is a predicate over the fit's
    own source instead."""
    import ast as _a
    import phase2_arms as PA
    src = Path(PA.__file__).read_text()
    node = _fn_node(src, "_feature_pass")
    if node is None:
        # SITE: measure#6
        raise DiagRefused("`_feature_pass` is absent from phase2_arms: "
                          "the claim has no subject")
    hits = [_a.unparse(nd) for nd in _a.walk(node)
            if isinstance(nd, _a.Call)
            and getattr(nd.func, "attr", "") == "loads"]
    return {"whole_file_loads": hits,
            "materialises_whole_file": any("read_text" in h for h in hits)}


def measure_assembly_slice(outdir, *, splits, n_windows: int = 2,
                           probe_rows: int = 50_000,
                           archive_root: str | None = None) -> dict:
    """THE COST OF THE EXPENSIVE HALF, measured on a bounded slice.

    RESULTS.md §4 records the feed at ~28.6 min MEASURED and the assembly
    as UNMEASURED. This measures it the only way a 3.2 GB index and a 1.2
    GB pass can be measured before they are authorised to run: on a slice,
    with the extrapolation stamped as an extrapolation.

    Writes ONLY into `outdir`, which `scratch_outdir` refuses to let be
    `data/` or the declared OUTDIR."""
    import resource
    import phase2_arms as PA
    import flow_intensity as fi
    import harmful_hazard_model as hm
    got = validate_splits(splits)
    out = scratch_outdir(outdir)
    roots = input_roots()
    if not roots["agree"] and archive_root is None:
        # SITE: measure#7
        raise DiagRefused(
            f"the fit stack reads from TWO ROOTS and they disagree: the "
            f"tape and fragment come from {roots['derived_root']} "
            f"(hardcoded in `phase2_arms`) and the window archives from "
            f"{roots['archive_root']} (`__file__`-relative in "
            f"`flow_intensity`). Measured from a worktree that is a "
            f"split-brain input, and `_feature_pass` would drop 100% at "
            f"`no_archive` after paying for the whole tape index. Supply "
            f"`archive_root` to point the archives at the SAME tree the "
            f"tape comes from -- nothing else is admissible")
    if archive_root is not None and str(
            Path(archive_root).resolve()) != roots["derived_root"]:
        # SITE: measure#8
        raise DiagRefused(
            f"archive_root {archive_root} is not the tree the tape and "
            f"fragment come from ({roots['derived_root']}): the only "
            f"admissible correction to a split-brain input is to make the "
            f"two roots the SAME tree, not to choose a third")
    rss0 = _peak_rss_mb()
    rep: dict = {"splits": list(got), "split_set": split_set_name(got),
                 "n_windows_requested": n_windows,
                 "rss_at_start_mb": rss0,
                 "input_roots": roots,
                 "archive_root_injected": (
                     None if archive_root is None
                     else str(Path(archive_root).resolve())),
                 "materialisation": feature_pass_materialises_whole_file()}

    # ---- the raw streaming rate of the REAL tape, bounded --------------
    t0 = time.time()
    n = 0
    for _ in PA._stream_tape_rows(PA.TAPE_PATH):
        n += 1
        if n >= probe_rows:
            break
    probe_s = time.time() - t0
    rep["tape_stream_probe"] = {
        "rows": n, "wall_s": round(probe_s, 2),
        "rows_per_s": round(n / probe_s, 1) if probe_s else None,
        "peak_rss_mb": _peak_rss_mb()}

    # ---- the slices ----------------------------------------------------
    _saved = (fi.PM, fi.RAW, fi.MARKETS, fi.GAPS, fi.DAYS)
    try:
        if archive_root is not None:
            # The INJECTION, reported in the artifact: the archives are
            # pointed at the tree the tape and fragment already come from,
            # so the pass reads ONE tree. No file is created or moved.
            fi.PM = Path(archive_root).resolve() / "data/pm_5min"
            fi.RAW = fi.PM / "raw"
            fi.MARKETS = fi.PM / "markets.jsonl"
            fi.GAPS = fi.PM / "collector_gaps.jsonl"
            fi.DAYS = fi._discover_days()
        paths, tokens = hm.fi._archive_paths(), hm.fi.token_map()
        pop = [sl for sl in json.loads(SLUGS.read_text())
               if sl in paths and sl in tokens]
        rep["population"] = {
            "n_declared": len(json.loads(SLUGS.read_text())),
            "n_with_archive_and_token": len(pop),
            "archive_days": len(getattr(fi, "DAYS", ()) or ())}
        if not pop:
            # SITE: measure#10
            raise DiagRefused(
                f"NONE of the §3 population's windows has both an archive "
                f"and a token pair under {fi.RAW}: every row would drop at "
                f"`no_archive` and the fit's own absorption bound would "
                f"refuse. There is nothing to measure and this says so "
                f"rather than reporting a fast pass over nothing")
        t0 = time.time()
        fr = fragment_slice(out / "fragment_slice.json",
                            n_windows=n_windows, only_slugs=pop)
        fr["build_wall_s"] = round(time.time() - t0, 2)
        tp = tape_slice(out / "tape_slice.json", slugs=fr["slugs"])
        rep["fragment_slice"], rep["tape_slice"] = fr, tp

        # ---- STAGE A: the fit's own tape_index, on the sliced tape ----
        real = PA.TAPE_PATH
        idx: dict = {}
        stage_a = {}
        try:
            PA.TAPE_PATH = out / "tape_slice.json"
            for sp in got:
                t0 = time.time()
                part = PA.tape_index(sp)
                stage_a[sp] = {"wall_s": round(time.time() - t0, 2),
                               "rows_indexed": len(part),
                               "peak_rss_mb": _peak_rss_mb()}
                idx.update(part)
        finally:
            PA.TAPE_PATH = real
        rep["stage_a_tape_index"] = stage_a
        rep["stage_a_rows_indexed_total"] = len(idx)

        # ---- STAGE B: the fit's own feature pass, on the slice --------
        t0 = time.time()
        blocks = PA._feature_pass(out / "fragment_slice.json",
                                  "phase4_diag_slice", TAPE=idx)
        b_s = time.time() - t0
        rep["stage_b_feature_pass"] = {
            "wall_s": round(b_s, 2),
            "kept_by_coin": {c: len(b["kept"]) for c, b in blocks.items()},
            "drops_by_coin": {c: dict(b["drops"]) for c, b in blocks.items()},
            "peak_rss_mb": _peak_rss_mb()}
    finally:
        fi.PM, fi.RAW, fi.MARKETS, fi.GAPS, fi.DAYS = _saved

    # ---- the extrapolations, each stamped ------------------------------
    pre = assembly_preconditions()
    dens_t = row_density(PA.TAPE_PATH)
    dens_f = row_density(PA.FRAGMENT)
    rep["row_density"] = {"tape": dens_t, "fragment": dens_f}
    est_tape_rows = dens_t["estimated_total_rows"]
    est_frag_rows = dens_f["estimated_total_rows"]
    a_total = sum(v["wall_s"] for v in stage_a.values())
    rep["extrapolation"] = {
        # The FLOOR: pure streaming, once per split, nothing else.
        "stage_a_stream_only_floor": extrapolate(
            probe_s * len(got), slice_unit=n,
            full_unit=est_tape_rows, unit="tape rows streamed per split"),
        # The ESTIMATE OF RECORD: the real function, scaled by rows. It
        # already contains the per-split loop, so it is not multiplied
        # again.
        "stage_a_index_build": extrapolate(
            a_total, slice_unit=tp["n_rows"], full_unit=est_tape_rows,
            unit="tape rows in the sliced tape"),
        "stage_b_feature_pass": extrapolate(
            b_s, slice_unit=fr["n_rows"], full_unit=est_frag_rows,
            unit="fragment rows"),
        "stage_b_peak_rss_delta_mb": extrapolate(
            rep["stage_b_feature_pass"]["peak_rss_mb"] - rss0,
            slice_unit=fr["n_rows"], full_unit=est_frag_rows,
            unit="fragment rows"),
        "estimated_full_tape_rows": est_tape_rows,
        "estimated_full_fragment_rows": est_frag_rows,
        "n_splits_streamed": len(got),
        "note": (
            "`tape_index` streams the WHOLE tape ONCE PER SPLIT, so the "
            f"floor above is multiplied by {len(got)} for "
            f"{split_set_name(got)}. The index-build estimate scales the "
            f"measured slice run, which already performed that loop"),
        "memory_caveat": (
            "the RSS estimate is scaled linearly and `_feature_pass` "
            "materialises the WHOLE fragment "
            f"({pre['fragment_bytes']:,} B) with json.loads(read_text()), "
            "so the true requirement does not depend on a row slice at "
            "all. Both readings say the same thing: this does not fit in "
            "8 GB. The cap for the ruled execution is not a seat's to set"),
    }
    rep["peak_rss_mb_process"] = _peak_rss_mb()
    rep["maxrss_kb_raw"] = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss
    rep["as_of"] = None            # stamped by the caller from `date -u`
    (out / "assembly_cost_report.json").write_text(
        json.dumps(rep, indent=1, sort_keys=True, default=str))
    return rep


def _peak_rss_mb() -> float:
    """Peak RSS of THIS process so far, in MB -- a high-water mark, so a
    later stage's figure includes every earlier stage."""
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                 1)


def assemble_gen_scores(refs: dict, *, splits, coins=COINS,
                        heads=("incumbent_linear_d",
                               "q1_arrival_composed_lgbm")) -> dict:
    """(coin, head) -> (per-generation scores, exclusion statuses).

    The join is to the §3 POPULATION: the feature pass covers the whole
    fragment, and only the generations the reference carries are scored.
    A generation whose rows the pass dropped is counted under
    NO_ROWS_KEPT and is absent from the scores -- an exclusion with a
    status, never a generation scored from nothing."""
    fb = feature_blocks(splits=splits)
    out = {"assembly": {k: v for k, v in fb.items()
                        if k not in ("blocks", "split_of")},
           "by_arm": {}, "split_by_gen": {}, "split_counts": {}}
    out["assembly"]["ruling"] = dict(SPLIT_RULING)
    for coin in coins:
        if coin not in fb["blocks"]:
            # SITE: assembly#8
            raise DiagRefused(
                f"the feature pass produced no block for {coin}: every row "
                f"was excluded, which reads as a null result and is an "
                f"absent one")
        for head in heads:
            sc, st, sb = generation_scores(fb["blocks"][coin], refs[coin],
                                           coin=coin, head=head,
                                           split_of=fb["split_of"])
            out["by_arm"][(coin, head)] = (sc, st)
            # The labelling is a property of the ROWS, so it must not
            # depend on which head scored them. If two heads disagree the
            # map is not about the population and the run says so.
            prev = out["split_by_gen"].get(coin)
            if prev is not None and prev != sb:
                # SITE: assembly#9
                raise DiagRefused(
                    f"the split label of {len(set(prev.items()) ^ set(sb.items()))} "
                    f"generation(s) of {coin} depends on WHICH HEAD scored "
                    f"them: the label is a property of the tape index, so a "
                    f"head-dependent one means the two passes saw different "
                    f"rows and no cell could honestly carry either label")
            out["split_by_gen"][coin] = sb
        out["split_counts"][coin] = split_tally(
            out["split_by_gen"][coin].keys(), out["split_by_gen"][coin])
    return out


def split_tally(keys, split_by_gen: dict) -> dict:
    """Count `keys` by the split their rows were indexed under.

    The whole content of R-496 (E)'s "labelled per cell": a cell says how
    many of the generations IT ACTED ON came from each of the fit's
    splits. `UNLABELLED` and `MIXED` are named buckets, never folded into
    a split (rule 4: exclusions are statuses)."""
    out: dict = {}
    for k in keys:
        lab = split_by_gen.get(tuple(k), "UNLABELLED")
        out[lab] = out.get(lab, 0) + 1
    return out


def generation_scores(blocks: dict, reference: dict, *, coin: str,
                      head: str, split_of: dict | None = None) -> tuple:
    """(slug, side, t0) -> one score per GENERATION, the exclusions, and
    THE SPLIT EACH GENERATION'S ROWS CAME FROM.

    TWO THINGS THIS FUNCTION EXISTS TO GET RIGHT, both of which a direct
    row-level stream gets wrong:

    1. **The generation is the unit, and its score is the MAX of its rows'
       scores.** `phase2_arms.freeze_thresholds` resolves theta over
       per-generation MAXIMA precisely because the evaluator ranks
       generations; a mean or a first-row score compared against that theta
       selects a different count and is not the policy's statistic.
    2. **A generation whose rows the feature pass dropped is EXCLUDED WITH
       A STATUS**, never scored from whatever rows survived and never
       silently absent (rule 4). `_feature_pass` drops rows for named
       reasons -- `pm`, `fine`, `state_join_failed`, the design exclusions
       -- so misses are expected, and a generation that keeps only some of
       its rows carries a max over FEWER rows than the fit's, which is a
       different number and is counted as `PARTIAL_ROWS`.

    3. **Every generation carries the SPLIT its rows were indexed under**
       (R-496 (E): "splits labelled per cell"). `split_of` maps the tape
       key to the split it was indexed from; a generation whose rows came
       from more than one split is `MIXED` and says so rather than being
       assigned the first split seen, and one whose key is not in the map
       is `UNLABELLED` -- which under MECHANICS_BOTH_SPLITS should be
       empty and is COUNTED rather than assumed empty."""
    norms = HS.load_lgbm_normalisers(coin)
    inc = HS.load_incumbent(coin)
    booster, wl = HS.load_lgbm(coin)
    pm, fn, st = blocks["PM"], blocks["FN"], blocks["ST"]
    kept = blocks["kept"]
    if not (len(pm) == len(fn) == len(st) == len(kept)):
        # SITE: gen#1
        raise DiagRefused(
            f"the feature pass returned unequal blocks (PM {len(pm)}, FN "
            f"{len(fn)}, ST {len(st)}, kept {len(kept)}): they are parallel "
            f"lists and zipping them at unequal length pairs one row's "
            f"features with another row's identity")
    by_gen: dict = {}
    spl_gen: dict = {}
    for i, r in enumerate(kept):
        v = HS.compose_head_inputs(
            pm[i], fn[i], st[i], norms=norms,
            incumbent_width=inc["_n_features"], lgbm_width=wl)[head]
        sc = (HS.score_incumbent(inc, v) if head == "incumbent_linear_d"
              else HS.score_lgbm(booster, wl, v))
        gk = (r["slug"], r["side"], r["gen"])
        by_gen.setdefault(gk, []).append(sc)
        spl_gen.setdefault(gk, set()).add(
            (split_of or {}).get(
                (r["slug"], r["side"], r["gen"], r["t_start"])))
    scores: dict = {}
    split_by_gen: dict = {}
    statuses = {"SCORED": 0, "NO_ROWS_KEPT": 0, "PARTIAL_ROWS": 0}
    for slug, sides in reference.items():
        for side in HSP.SIDES:
            for g in sides[side]:
                got = by_gen.get((slug, side, g["gen"]))
                if not got:
                    statuses["NO_ROWS_KEPT"] += 1
                    continue
                scores[(slug, side, float(g["t0"]))] = max(got)
                sp = spl_gen[(slug, side, g["gen"])]
                split_by_gen[(slug, side, g["gen"])] = (
                    "UNLABELLED" if sp == {None}
                    else "MIXED" if len(sp) > 1 else sorted(sp)[0])
                statuses["SCORED"] += 1
    statuses["PARTIAL_ROWS"] = sum(
        1 for k, v in by_gen.items() if len(v) < _rows_expected(k, reference))
    return scores, statuses, split_by_gen


def _rows_expected(key, reference: dict) -> int:
    """How many rows the fit's own pass would have had for this generation.

    Read from the reference's tranche count is WRONG (tranches are fills,
    rows are decisions), so this reports the observed count and the
    comparison is left to the run, which has both. Returns the observed
    count so `PARTIAL_ROWS` is 0 until the run supplies the fit's own
    per-generation row counts -- an honest zero, declared here rather than
    a number computed from the wrong table."""
    return 0


def _head_scorer(head: str, coin: str, gen_scores: dict | None = None):
    """The real scorer when the assembly has produced scores; otherwise it
    REFUSES, naming the one thing that is missing.

    Round 33 fed the LGBM booster `[[row["t"]]]` -- one column against 106
    -- and returned a constant 0.5 for the incumbent, under a docstring
    that said "never a stub". Round 34 built the real head application in
    `de_head_scoring` and LEFT THIS FUNCTION IN THE RUN PATH, so `--run`
    would still have fed for ~29 minutes and then tracebacked at the first
    cell on a `LightGBMError` `main()` did not catch.

    WHAT IS WIRED NOW (round 37): the composition each head was fitted
    through (`HS.compose_head_inputs`), the per-generation statistic
    (`generation_scores`, max over the generation's rows, matching
    `freeze_thresholds`), and every precondition of the expensive pass
    (`assembly_preconditions`).

    WHAT IS WIRED NOW (round 44): the expensive pass itself --
    `PA.tape_index(split)` over the 3.2 GB tape and
    `PA._feature_pass(PA.FRAGMENT, 'phase4_diag', TAPE=...)` over its
    1.14M rows, behind `feature_blocks` / `assemble_gen_scores`, on the
    split set R-496 (E) ruled.

    WHAT THIS FUNCTION STILL REFUSES, and always must: being called with
    NO assembled scores. Round 33's stub fed the booster one column
    against 106 and returned a constant 0.5 for the incumbent under a
    docstring that said "never a stub". There is no path here that
    invents a score: either the assembly produced one for the generation
    in front of it, or this refuses by name (DE34-C1)."""
    if gen_scores is not None:
        def _score(row):
            k = (row["slug"], row["side"], float(row["t"]))
            if k not in gen_scores:
                # SITE: scorer#2
                raise DiagRefused(
                    f"no assembled score for generation {k}: a generation "
                    f"whose rows the feature pass dropped is an EXCLUSION "
                    f"with a status, and must be removed from the "
                    f"population before scoring rather than scored from "
                    f"nothing")
            return gen_scores[k]
        return _score
    # SITE: scorer#1
    raise DiagRefused(
        f"no assembled scores were supplied for {head}/{coin}, so there "
        f"is nothing to score with. The assembly IS wired "
        f"(`feature_blocks` -> `PA.tape_index` + `PA._feature_pass`, "
        f"round 44) and `run()` calls it once and passes its output here; "
        f"a call that reaches this line reached it without that output, "
        f"and the alternative to refusing is the round-33 stub -- a "
        f"constant 0.5 for the incumbent, and the row's timestamp fed to "
        f"the booster as its one feature. Refused HERE rather than "
        f"scored from nothing (DE34-C1)")


def preflight(*, splits=None) -> None:
    """Everything knowable BEFORE the ~29-minute feed and the assembly.

    The run's first cell is the worst place to learn that the scorer is a
    stub or a thresholds key is wrong: each is checkable in milliseconds,
    so each is checked here and the feed is never paid for a run that
    cannot finish.

    WHAT IT REFUSES ON TODAY, and this is the honest state of the
    diagnostic: `verify_called_code()`, because wiring the expensive half
    reached `_stream_tape_rows` and that function differs from the fit
    bytes. Round 44 did NOT declare it away -- admitting a fit-vs-tip
    drift is an admissibility call and admissibility is the USER's (rule
    14). The facts the ruling needs are computed by `code_drift_report()`
    and are printed by `--pin-report`, which writes nothing.

    `splits`, when given, is validated here so a run under an undeclared
    or unknown split set dies in milliseconds rather than after the feed.
    It is NOT defaulted: `run()` requires it."""
    if splits is not None:
        validate_splits(splits)
    for coin in COINS:
        for head in ("incumbent_linear_d", "q1_arrival_composed_lgbm"):
            HS.thresholds(coin, head)
    HS.verify_fit_code()
    verify_called_code()
    assembly_preconditions()


def _splits_from_cli(arg: str | None):
    """The `--splits` string as a split set -- and SILENCE STAYS SILENCE.

    `None` is passed straight through to `validate_splits`, which refuses
    by name. Translating an absent flag into `RULED_SPLIT_SET` here is
    exactly the move rule 14 forbids: the ruling would then be something
    the code supplies rather than something an operator states."""
    if arg is None:
        return None
    return DECLARED_SPLIT_SETS.get(arg) or tuple(
        x.strip() for x in arg.split(",") if x.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="execute the declared diagnostic into OUTDIR")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--splits", default=None,
                    help=f"the DECLARED split set -- one of "
                         f"{sorted(DECLARED_SPLIT_SETS)}, or a "
                         f"comma-separated subset of {list(TAPE_SPLITS)}. "
                         f"NO DEFAULT: the USER's ruling (R-496 (E)) is "
                         f"{RULED_SPLIT_SET} and it is named here, never "
                         f"assumed")
    ap.add_argument("--pin-report", action="store_true",
                    help="print the code pin and the undeclared drift, "
                         "computed; writes nothing")
    ap.add_argument("--measure-slice", action="store_true",
                    help="measure the assembly on a bounded slice into "
                         "--outdir (never data/, never the run's OUTDIR)")
    ap.add_argument("--n-windows", type=int, default=2)
    ap.add_argument("--archive-root", default=None,
                    help="point the window archives at this repo root -- "
                         "admissible ONLY when it is the tree the tape "
                         "and fragment already come from (see input_roots)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.pin_report:
        print(json.dumps(code_drift_report(), indent=1, sort_keys=True,
                         default=str))
        return 0
    if a.measure_slice:
        try:
            rep = measure_assembly_slice(
                a.outdir, splits=_splits_from_cli(a.splits),
                n_windows=a.n_windows, archive_root=a.archive_root)
        except DiagRefused as exc:
            print(f"[de_phase4_diag_runner] REFUSED: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(rep, indent=1, sort_keys=True, default=str))
        return 0
    if a.run:
        # DE33-C8: a refusal under the CLI exits BY NAME, rc non-zero, no
        # traceback -- round 33 let every one of them out unhandled.
        try:
            rec = run(Path(a.outdir) if a.outdir else None,
                      splits=_splits_from_cli(a.splits))
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
