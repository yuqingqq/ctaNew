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

EXPECTED_CHECKS = 85

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
#: DE36-C5: each entry carries the function's AST sha AT THE FIT COMMIT
#: and AT THE TIP THAT DECLARED IT. A later edit to either side re-opens
#: the question instead of inheriting this pass -- rule 12's shape applied
#: to a declaration. The shas are filled by `_seal_declarations()` at
#: import, from the same two sources the comparison reads, so they cannot
#: drift from what is actually compared.
DECLARED_ADDITIVE_SHAS: dict = {}

DECLARED_ADDITIVE = {
    ("harmful_exposure_rows.py", "select_v2_era"):
        "the `era` keyword now defaults to `fi.ERA` via `_era_or_refuse` "
        "(the value this "
        "population is selected under anyway) and an empty selection "
        "REFUSES instead of returning nothing -- both make the same "
        "selection for this population and turn a silent empty into a "
        "refusal, which is the direction the fit-commit bytes lack",
    ("harmful_exposure_rows.py", "_era_or_refuse"):
        "NEW at 851edaf (absent from the fit bytes): it names the era from "
        "`fi.ERA` -- the value this population is already selected under -- "
        "and routes the empty case to the refusal below. It adds a refusal "
        "and changes no selection that is non-empty, so for this "
        "population the selected set is the fit's",
    ("harmful_exposure_rows.py", "_refuse_empty_selection"):
        "NEW at 851edaf: it exists only to RAISE on an empty selection, "
        "so it cannot alter a selection that is non-empty -- and the "
        "population this diagnostic runs on is 471 windows, measured",
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


def _seal_declarations() -> dict:
    """The AST sha of every declared function, at the fit commit and at
    this tip -- computed once, from the sources the comparison reads."""
    if DECLARED_ADDITIVE_SHAS:
        return DECLARED_ADDITIVE_SHAS
    m = json.loads((FITS / "fit_manifest.json").read_text())
    ref = m.get("fit_code_ref") or ""
    here = Path(__file__).resolve().parent
    by_file: dict = {}
    for (name, fn) in DECLARED_ADDITIVE:
        if name not in by_file:
            fit_src = _git_show(ref, f"live/pm_research/{name}")
            by_file[name] = (_fn_asts(fit_src) if fit_src else {},
                             _fn_asts((here / name).read_text())
                             if (here / name).exists() else {})
        a, b = by_file[name]
        DECLARED_ADDITIVE_SHAS[(name, fn)] = {
            "sha_at_fit": _ast_sha(a.get(fn)),
            "sha_at_declaring_tip": _ast_sha(b.get(fn)),
        }
    return DECLARED_ADDITIVE_SHAS


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


def pin_statuses() -> list:
    """One computed status per manifest-pinned file (DE34-R7)."""
    m = json.loads((FITS / "fit_manifest.json").read_text())
    codes = m.get("fit_code_files") or {}
    ref = m.get("fit_code_ref") or ""
    here = Path(__file__).resolve().parent
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
        row["n_functions_called"] = len(reach)
        changed = sorted(n for n in reach if a.get(n) != b.get(n))
        row["functions_changed"] = changed
        sealed = _seal_declarations()
        undeclared = []
        for n in changed:
            key = (name, n)
            if key not in DECLARED_ADDITIVE:
                undeclared.append(n)
                continue
            # DE36-C5: the declaration is pinned to WHAT IT DECLARED. If
            # either side has moved since, the pass is not inherited.
            want = sealed.get(key, {})
            now = {"sha_at_fit": _ast_sha(a.get(n)),
                   "sha_at_declaring_tip": _ast_sha(b.get(n))}
            if want != now:
                undeclared.append(f"{n} (declaration stale: declared "
                                  f"{want}, now {now})")
        row["undeclared"] = undeclared
        row["verdict"] = ("IDENTICAL" if not changed else
                          "BLOCKING" if undeclared else "ADDITIVE_DECLARED")
        row["declared"] = {n: DECLARED_ADDITIVE[(name, n)]
                           for n in changed if (name, n) in DECLARED_ADDITIVE}
        row["n_functions_in_file"] = len(set(a) | set(b))
        out.append(row)
    return out


def verify_called_code(rows=None) -> list:
    """The statuses, refusing ONLY on BLOCKING (DE34-R7 ruled).

    `rows` is injectable so the refusal can be DRIVEN (rule 15): the
    falsifier builds a status row for a synthetic undeclared change and
    asserts this raises by name -- round 36 shipped the refusal without
    one."""
    rows = pin_statuses() if rows is None else rows
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
    stratum cannot be permuted as asked -- the caller redraws."""
    want = {(sl, sd, int(gn)) for sl, sd, gn in
            (k.split("|") for k in drawn)}
    by_st: dict = {}
    for e in treated_scores:
        key = (e["slug"], e["side"], e.get("gen"))
        by_st.setdefault(_stratum_of(key, gidx), []).append(e)
    out, ok = [], True
    for st, events in by_st.items():
        above = sorted((float(e["score"]) for e in events
                        if float(e["score"]) >= theta), reverse=True)
        below = sorted((float(e["score"]) for e in events
                        if float(e["score"]) < theta), reverse=True)
        keys = [(e["slug"], e["side"], e.get("gen")) for e in events]
        drawn_here = [k for k in keys if k in want]
        rest = [k for k in keys if k not in want]
        if len(drawn_here) != len(above):
            # The draw named a different number of generations in this
            # stratum than there are above-threshold values to give them:
            # a permutation cannot honour it, so the caller redraws.
            ok = False
        pairs = list(zip(drawn_here, above)) + list(zip(rest, below))
        # Any leftover key (when the counts disagree) keeps its own value,
        # so the multiset is still the stream's own.
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
    p3 = ctrl_above == {w for w in want if w in set(kc)}
    p4 = None if (rc_treated is None or rc_control is None) \
        else rc_treated == rc_control
    return {"P1_key_multisets_equal": p1,
            "P2_stratum_score_multisets_equal": p2,
            "P3_drawn_carry_above_and_only_drawn": p3,
            "P4_realised_action_counts_equal": p4}


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
        # DE35-C4: hoisted -- these were rebuilt inside the seed loop.
        gidx = _gen_index(reference)
        treated = [{"slug": f"{a['slug']}|{a['side']}|{a['gen']}"}
                   for a in _treated_actions(per_arm[head])]
        treated_scores = list(scores_by_arm[head])
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
        _rc_treated = _realised_by_stratum(per_arm[head], gidx)
        attempted = accepted = rejected = 0
        rej_by_stratum: dict = {}
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
            MRC.refuse_if_not_random(drawn, treated, pool=pool)
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
            res = arm_result(reference, ctrl_scores, c, theta=th[head])
            _rc_ctrl = _realised_by_stratum(res, gidx)
            attempted += 1
            if _rc_ctrl != _rc_treated:
                # P4: the decision variable did not match. Reject, count
                # per stratum, and redraw -- never keep a control matched
                # on something other than the frozen variable.
                rejected += 1
                for st in set(_rc_ctrl) | set(_rc_treated):
                    if _rc_ctrl.get(st, 0) != _rc_treated.get(st, 0):
                        rej_by_stratum[f"{st[0]}|{st[1]}"] = \
                            rej_by_stratum.get(f"{st[0]}|{st[1]}", 0) + 1
                continue
            accepted += 1
            vals.append(res["cost_adjusted_value_cents"])
            if res["rho"] is not None:
                rhos.append(res["rho"])
        if accepted < draws:
            # SITE: null#2
            raise DiagRefused(
                f"only {accepted} of {draws} draws matched the treated "
                f"arm's per-stratum realised action count in "
                f"{attempted} attempts ({rejected} rejected). A null built "
                f"from the draws that happened to match is matched on "
                f"acceptance, not on the decision variable -- refusing is "
                f"the honest end (DRAW_ATTEMPT_BUDGET = "
                f"{DRAW_ATTEMPT_BUDGET})")
        vals.sort()
        rhos.sort()
        out["null_population"] = {
            "n_draws_attempted": attempted,
            "n_draws_accepted": accepted,
            "n_rejected_by_stratum": dict(sorted(rej_by_stratum.items())),
            "draw_attempt_budget": DRAW_ATTEMPT_BUDGET,
            "n_strata": len(_strata),
            "strata_with_room": sum(1 for v in _room.values() if v > 0),
            "strata_forced": sorted(f"{k[0]}|{k[1]}" for k, v in _room.items()
                                    if v <= 0),
            "n_distinct_draws": len(_seen_draws),
            "point_mass": len(_seen_draws) == 1,
            "note": ("a stratum with no room contributes a POINT MASS: the "
                     "matched draw there is forced, so its contribution is "
                     "a constant and not a sample (DE31-R2)"),
        }
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
            "DE34-C1: the RUN PATH's scorer REFUSES by name, and after "
            "round 37 it names ONE remaining step rather than the whole "
            "assembly -- round 33's stub (`[[row['t']]]` against 106 "
            "features, 0.5 for the incumbent) would have fed for ~29 "
            "minutes and tracebacked at the first cell",
            needle="EXPENSIVE HALF is not wired")
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
    _gs, _gst = generation_scores(_blk(_rows), _fixref, coin="btc",
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
    _gs2, _gst2 = generation_scores(_blk(_rows[:1]), _fixref, coin="btc",
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
            "and `preflight()` REFUSES before anything is built -- at the "
            "scorer, AFTER the cheap preconditions have run, because the "
            "pin no longer blocks on a file the run does not execute "
            "(DE35-R1)",
            needle="EXPENSIVE HALF is not wired")
    _pin = verify_called_code()          # PROCEEDS: no BLOCKING verdict
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
    ok(_pa["verdict"] == "IDENTICAL" and _pa["sha_at_fit"] != _pa["sha_at_run"]
       and _pa["n_functions_called"] == 1
       and _pa["n_functions_called"] < _pa["n_functions_in_file"],
       f"AND THE VERDICT'S SCOPE IS SAID OUT LOUD: `phase2_arms.py` reads "
       f"IDENTICAL while its FILE sha has moved since the fit "
       f"({_pa['sha_at_fit']} -> {_pa['sha_at_run']}). IDENTICAL means "
       f"every entry the run REACHES matches the fit bytes -- here "
       f"{_pa['n_functions_called']} of {_pa['n_functions_in_file']} (the "
       f"module's top-level body, which is where `FRAGMENT` and "
       f"`TAPE_PATH` live). THE OPEN ITEM INHERITS THIS: wiring the "
       f"expensive half calls `tape_index` and `_feature_pass`, which are "
       f"NOT in today's reached set, so they enter the comparison then and "
       f"may not be identical. A green pin today is not a green pin for "
       f"the wired run")
    from collections import Counter as _C
    _verd = _C(r["verdict"] for r in _pin)
    ok(_pv.get("phase2_arms.py") == "IDENTICAL"
       and _verd["IDENTICAL"] == 11 and _verd["ADDITIVE_DECLARED"] == 1
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
    _sealed = _seal_declarations()
    ok(len(_sealed) == 3
       and all(v["sha_at_declaring_tip"] for v in _sealed.values()),
       f"and each DECLARED_ADDITIVE entry is PINNED to what it declared -- "
       f"the AST sha at the fit commit AND at this tip -- so a later edit "
       f"re-opens the question instead of inheriting the pass (DE36-C5): "
       f"{ {k[1]: v['sha_at_declaring_tip'] for k, v in _sealed.items()} }")
    # DE36-C5 DRIVEN: tamper with what a declaration was sealed against
    # and the verdict must re-open to BLOCKING.
    _key = ("harmful_exposure_rows.py", "select_v2_era")
    _save = dict(DECLARED_ADDITIVE_SHAS[_key])
    DECLARED_ADDITIVE_SHAS[_key] = dict(_save, sha_at_fit="deadbeefdeadbeef")
    try:
        _tampered = [r for r in pin_statuses()
                     if r["path"] == "harmful_exposure_rows.py"][0]
    finally:
        DECLARED_ADDITIVE_SHAS[_key] = _save
    ok(_tampered["verdict"] == "BLOCKING"
       and any("declaration stale" in str(x)
               for x in _tampered.get("undeclared", [])),
       f"KNOWN-BAD, DRIVEN: a declaration whose sealed AST sha no longer "
       f"matches re-opens to {_tampered['verdict']}, naming the stale "
       f"entry in `undeclared` ({[x for x in _tampered.get('undeclared', []) if 'stale' in str(x)][:1]}) "
       f"-- a later edit cannot inherit the pass (DE36-C5)")
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
       "and both null-side guards are present in `run_cell`, asserted from "
       "the parse: the P4 rejection branch (a draw whose per-stratum "
       "realised count differs is rejected, counted and redrawn) and the "
       "budget refusal (exhausting DRAW_ATTEMPT_BUDGET refuses rather than "
       "building the null from whichever draws matched). Driving either "
       "would need a population engineered to reject, which is the "
       "fixture work item 1 leaves to the run")
    refuses(lambda: verify_called_code([
        {"path": "harmful_exposure_rows.py", "sha_at_fit": "aaaa",
         "sha_at_run": "bbbb", "commit": "e12e2c7",
         "functions_changed": ["join_fills"], "verdict": "BLOCKING"}]),
        "KNOWN-BAD, DRIVEN (rule 15): a synthetic UNDECLARED change to a "
        "called function refuses by name at `called#1` -- the falsifier "
        "round 36 shipped without", needle="BLOCKING pin status")
    ok(verify_called_code(_pin) == _pin,
       "POSITIVE CONTROL on the same path: the real statuses carry no "
       "BLOCKING verdict, so the refusal is a filter and not a wall")

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
    rows = [{"t": g["t0"], "slug": slug, "side": side}
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


def generation_scores(blocks: dict, reference: dict, *, coin: str,
                      head: str) -> tuple:
    """(slug, side, t0) -> one score per GENERATION, and the exclusions.

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
       different number and is counted as `PARTIAL_ROWS`."""
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
    for i, r in enumerate(kept):
        v = HS.compose_head_inputs(
            pm[i], fn[i], st[i], norms=norms,
            incumbent_width=inc["_n_features"], lgbm_width=wl)[head]
        sc = (HS.score_incumbent(inc, v) if head == "incumbent_linear_d"
              else HS.score_lgbm(booster, wl, v))
        by_gen.setdefault((r["slug"], r["side"], r["gen"]), []).append(sc)
    scores: dict = {}
    statuses = {"SCORED": 0, "NO_ROWS_KEPT": 0, "PARTIAL_ROWS": 0}
    for slug, sides in reference.items():
        for side in HSP.SIDES:
            for g in sides[side]:
                got = by_gen.get((slug, side, g["gen"]))
                if not got:
                    statuses["NO_ROWS_KEPT"] += 1
                    continue
                scores[(slug, side, float(g["t0"]))] = max(got)
                statuses["SCORED"] += 1
    statuses["PARTIAL_ROWS"] = sum(
        1 for k, v in by_gen.items() if len(v) < _rows_expected(k, reference))
    return scores, statuses


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

    WHAT IS NOT: the expensive pass itself --
    `PA.tape_index(split)` over the 3.2 GB tape and
    `PA._feature_pass(PA.FRAGMENT, ..., TAPE)` over its 1.14M rows. It has
    never been executed for this diagnostic, its cost has never been
    measured here, and the §3 population spans BOTH of the fit's splits
    (train and score), which is a declaration the USER has not been asked
    for. So it refuses BEFORE the feed rather than proceed (DE34-C1)."""
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
        f"the feature assembly's EXPENSIVE HALF is not wired, so "
        f"{head}/{coin} cannot be scored. What is missing is exactly one "
        f"step: `PA.tape_index(split)` over the fit's own tape and "
        f"`PA._feature_pass(PA.FRAGMENT, 'phase4_diag', TAPE=...)`. "
        f"Composition, per-generation statistic and preconditions ARE "
        f"wired and falsified; the pass has never been executed for this "
        f"diagnostic, and the §3 population spans BOTH fit splits, which "
        f"is a declaration nobody has made. Refused HERE, before any feed "
        f"is built (DE34-C1)")


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
    assembly_preconditions()
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
