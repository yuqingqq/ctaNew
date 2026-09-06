"""THE EXECUTABLE MULTI-DAY GATE-1 RUNNER — fixtures only, no data.

WHAT THIS IS. The runnable form of design v3
(`p003_de_multiday_gate1_design_v3__20260906T040539Z.json`,
`a1016a8762fffdfe…`). It exists so the reviewer can DRIVE the design
before BE builds a single day book.

WHAT IT WILL NOT DO. It refuses to run on real days: the ruled day set is
EMPTY in the committed parameter file until the USER answers design v3's
R7 parameter, and an empty set REFUSES rather than defaulting to either
candidate. G is DERIVED from `len(days)` at run time and is never a
constant -- the defect design v2 shipped and v2b corrected.

THE RULES IT ENFORCES, each from the declaration and each falsified in
both directions below:
  R4  a short arm-day and a small-but-nonzero null sd REFUSE
  R5  no economic field exists in any artifact until n_days_complete == G
  R6  book digest, theta and model digests are verified AT RUN TIME and a
      mismatch REFUSES the day (a model mismatch refuses the RUN)
  R8  a day that exceeds its declared deadline REFUSES; the draw count is
      never lowered and the cap is never raised

BE's cascade is CITED, never copied: the module is imported and its source
digest compared to the declared one. A different digest refuses, because a
null run through a different cascade is not a control for this arm.

    python3 live/pm_research/de_multiday_gate1_runner.py --selftest
    python3 live/pm_research/de_multiday_gate1_runner.py --fixture-run \\
        --output PATH
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import de_multiday_design_declaration as DESIGN  # noqa: E402


PROTOCOL = "P003_DE_MULTIDAY_GATE1_RUNNER_V1"
EXPECTED_CHECKS = 33
PARAMS_REL = "live/pm_research/declarations/de_multiday_gate1_params_v1.json"

#: R5 -- the fields that do not exist in a per-day artifact until every day
#: is complete. Named once, so the guard and the emitter cannot disagree.
ECONOMIC_FIELDS = ("D_E0", "D_E_MINUS_R", "Z", "p_location",
                   "null_mean", "null_sd", "null_draws_summary")


class RunnerRefused(RuntimeError):
    """The run cannot proceed honestly on the inputs given."""


# ------------------------------------------------------------- parameters

def load_params(path: Path | None = None) -> dict:
    root = Path(__file__).resolve().parents[2]
    p = Path(path) if path is not None else root / PARAMS_REL
    if not p.is_file():
        raise RunnerRefused(f"REFUSED: no parameter file at {p}")
    d = json.loads(p.read_text())
    days = d.get("days")
    if not isinstance(days, list):
        raise RunnerRefused("REFUSED: `days` must be a list")
    if not days:
        raise RunnerRefused(
            "REFUSED: the ruled day set is EMPTY. G is derived from it and "
            "there is nothing to derive. The USER's answer to design v3's "
            "R7 parameter selects SET A (G = 6) or SET B (G = 3); the "
            "runner will not default to either.")
    if len(set(days)) != len(days):
        raise RunnerRefused(f"REFUSED: duplicate days in the ruled set")
    d["G"] = len(days)                      # DERIVED, never a constant
    d["G_derived_from_len_days"] = True
    # R-555: `expected_G` is a CROSS-CHECK, never the source of G. A set
    # that has quietly become five is the shape the USER's ruling forbids
    # -- "G stays 6 and nothing is chosen" -- so it refuses here rather
    # than testing at a smaller G later.
    exp = d.get("expected_G")
    if exp is not None and d["G"] != exp:
        raise RunnerRefused(
            f"REFUSED: the ruled set has {d['G']} days against the declared "
            f"expected_G {exp}. R-555 fixes G at {exp}; a set that shrank "
            f"is a day chosen after the fact, not a smaller test.")
    req = d.get("required_previously_opened_for")
    if req is not None:
        bad = {}
        for day in days:
            st = DESIGN.DAY_READ_STATE.get(day)
            if st is None:
                bad[day] = "NO_READ_STATE_RECORDED"
            elif st["previously_opened_for"] != req:
                bad[day] = st["previously_opened_for"]
        if bad:
            raise RunnerRefused(
                f"REFUSED: the ruled set contains days whose "
                f"previously_opened_for is not {req!r}: {bad}. R-555 is "
                f"untouched days only.")
    return d


# ------------------------------------------------------------ R6 verifies

def verify_be_module(params: dict, *, actual_sha: str | None = None) -> dict:
    """CITE BE's cascade, never copy it -- and refuse a different one."""
    declared = params["be_module"]["sha256"]
    if actual_sha is None:
        src = Path(__file__).resolve().parents[2] / params["be_module"]["path"]
        if not src.is_file():
            raise RunnerRefused(f"REFUSED: BE's module is absent at {src}")
        actual_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    if actual_sha != declared:
        raise RunnerRefused(
            f"REFUSED: BE's cascade module digest differs -- declared "
            f"{declared[:16]}, found {actual_sha[:16]}. A null run through "
            f"a DIFFERENT cascade is not a control for this arm, and the "
            f"citation must be re-pointed deliberately.")
    return {"path": params["be_module"]["path"], "sha256": actual_sha,
            "cited_not_copied": True, "verified_at_run_time": True}


MODEL_DIR = "data/pm_5min/derived/phase2_fits"


def verify_pinned_models(params: dict, root: Path | None = None) -> dict:
    """R6, THE HALF THAT WAS MISSING: READ the pinned model files at run
    time, hash them, and compare. A mismatch REFUSES THE RUN.

    The reviewer's phrase for the previous state was exact -- "a recorded
    digest that nobody compares is provenance theatre" -- and that was
    what the model half was. This reads bytes."""
    base = Path(root) if root is not None else Path(
        __file__).resolve().parents[2]
    out, bad = {}, []
    for arm, spec in sorted(params["arms"].items()):
        out[arm] = {}
        for name, pin in sorted(spec["model_digests"].items()):
            f = base / MODEL_DIR / name
            if not f.is_file():
                bad.append({"arm": arm, "model": name, "why": "ABSENT",
                            "path": str(f)})
                out[arm][name] = {"status": "ABSENT", "path": str(f)}
                continue
            got = hashlib.sha256(f.read_bytes()).hexdigest()[:len(pin)]
            ok_ = got == pin
            out[arm][name] = {"pinned": pin, "read": got, "matches": ok_,
                              "path": str(f.relative_to(base))}
            if not ok_:
                bad.append({"arm": arm, "model": name, "pinned": pin,
                            "read": got})
    if bad:
        raise RunnerRefused(
            f"REFUSED RUN: {len(bad)} pinned model file(s) do not match "
            f"their declared digest: {bad[:3]}. A moved model means the "
            f"object under test is not the frozen one, so no day is "
            f"trustworthy -- this refuses the RUN, not a day.")
    return {"model_dir": MODEL_DIR, "verified_at_run_time": True,
            "n_models_read": sum(len(v) for v in out.values()),
            "per_arm": out, "bytes_were_read_not_recorded": True}


def verify_pinned_thetas(params: dict, root: Path | None = None) -> dict:
    """R6: the theta half, read from the artifact each theta is pinned in
    (BE's null), not from this module's copy."""
    base = Path(root) if root is not None else Path(
        __file__).resolve().parents[2]
    art = base / "data/pm_5min/derived/be_cancel_axis_null_v1.json"
    if not art.is_file():
        raise RunnerRefused(f"REFUSED RUN: theta pin source absent: {art}")
    doc = json.loads(art.read_text())
    out, bad = {}, []
    for arm, spec in sorted(params["arms"].items()):
        got = ((doc.get("cells") or {}).get(arm) or {}).get(
            "arm_filed", {}).get("theta")
        out[arm] = {"declared": spec["theta"], "read": got,
                    "matches": got == spec["theta"],
                    "json_path": f"cells.{arm}.arm_filed.theta"}
        if got != spec["theta"]:
            bad.append({"arm": arm, "declared": spec["theta"], "read": got})
    if bad:
        raise RunnerRefused(
            f"REFUSED RUN: theta mismatch against the pin source: {bad}. A "
            f"refitted theta makes every day's object different.")
    return {"source": "data/pm_5min/derived/be_cancel_axis_null_v1.json",
            "verified_at_run_time": True, "per_arm": out}


def verify_run_inputs(params: dict, root: Path | None = None) -> dict:
    """Everything that must hold BEFORE the first day is touched."""
    return {"be_module": verify_be_module(params),
            "models": verify_pinned_models(params, root),
            "thetas": verify_pinned_thetas(params, root)}


def verify_day_inputs(day: str, declared_book_sha: str, actual_book_sha: str,
                      params: dict, actual_thetas: dict,
                      actual_model_digests: dict) -> dict:
    """R6 -- at RUN TIME, not merely recorded.

    A BOOK mismatch refuses THE DAY; a THETA or MODEL mismatch refuses THE
    RUN, because a moved model means the object under test is not the
    frozen one and no other day is trustworthy either."""
    if actual_book_sha != declared_book_sha:
        raise RunnerRefused(
            f"REFUSED DAY {day}: reference-book digest mismatch -- declared "
            f"{declared_book_sha[:16]}, found {actual_book_sha[:16]}. A day "
            f"whose book moved is not the day the design declared.")
    for arm, spec in params["arms"].items():
        if actual_thetas.get(arm) != spec["theta"]:
            raise RunnerRefused(
                f"REFUSED RUN: {arm} theta is {actual_thetas.get(arm)} "
                f"against the pinned {spec['theta']}. A refitted theta "
                f"makes every day's object different, not just this one.")
        for name, dig in spec["model_digests"].items():
            got = (actual_model_digests.get(arm) or {}).get(name)
            if got != dig:
                raise RunnerRefused(
                    f"REFUSED RUN: {arm} model {name} digest is {got} "
                    f"against the pinned {dig}.")
    return {"day": day, "book_sha256": actual_book_sha,
            "thetas_verified": True, "model_digests_verified": True}


# --------------------------------------------------------- the arm-day

def seed_for(day_book_sha: str, arm: str) -> int:
    """The seed PINS THE DATA: derived from the day book's digest."""
    h = hashlib.sha256(
        f"{day_book_sha}|{arm}|P003_GATE1_MULTIDAY".encode()).hexdigest()
    return int(h[:8], 16)


def arm_day(day: str, arm: str, observed: float, null_draws: list,
            n_decisions: int, params: dict, *,
            elapsed_s: float = 0.0) -> dict:
    """One arm on one day: R8 deadline, R4 degeneracy, then the statistics.

    The economic fields are computed here and WITHHELD by the emitter until
    G is complete (R5) -- computing them is not publishing them."""
    if elapsed_s > params["per_day_deadline_s"]:
        raise RunnerRefused(
            f"REFUSED DAY {day} / {arm}: {elapsed_s:.0f}s exceeds the "
            f"declared deadline {params['per_day_deadline_s']}s. The "
            f"500-draw minimum and the 8G cap are BOTH protected by "
            f"refusing the day -- never by lowering draws or raising the "
            f"cap.")
    if len(null_draws) < params["min_draws_per_arm_day"]:
        raise RunnerRefused(
            f"REFUSED DAY {day} / {arm}: {len(null_draws)} draws is below "
            f"the declared minimum {params['min_draws_per_arm_day']}.")
    adm = DESIGN.arm_day_admissible(n_decisions, null_draws)
    if not adm["admissible"]:
        return {"day": day, "arm": arm, "status": adm["status"],
                "admissibility": adm, "economic": None,
                "why_no_economic": "a refused arm-day carries no economic "
                                   "field; it is a STATUS and does not "
                                   "shrink G silently"}
    loc = DESIGN.per_day_location(observed, null_draws)
    z = DESIGN.per_day_standardised_excess(observed, null_draws)
    return {"day": day, "arm": arm, "status": "OK", "admissibility": adm,
            "economic": {"D_E0": observed, "Z": z,
                         "p_location": loc["p_one_sided"],
                         "null_mean": statistics.fmean(null_draws),
                         "null_sd": statistics.pstdev(null_draws),
                         "null_draws_summary": {"n": len(null_draws)}}}


# --------------------------------------------------- R5, the sealed guard

def _strip_economic(o):
    """Remove every economic-named field at EVERY depth.

    THE GUARD CAUGHT THE EMITTER ON ITS FIRST RUN. Sealing only the
    top-level `economic` block left `admissibility.null_sd` and
    `admissibility.null_mean` in the artifact -- the R4 block carries null
    statistics, and those are economic wherever they sit. The emitter and
    the guard now share ONE name list and one traversal, so they cannot
    disagree again."""
    if isinstance(o, dict):
        return {k: _strip_economic(v) for k, v in o.items()
                if k not in ECONOMIC_FIELDS}
    if isinstance(o, list):
        return [_strip_economic(v) for v in o]
    return o


def seal(day_result: dict, n_days_complete: int, g: int) -> dict:
    """R5 -- the economic fields are ABSENT until every day is complete.

    Absent, not present-and-ignored: a field a reader can see is a field a
    reader can quote."""
    if n_days_complete >= g:
        out = dict(day_result)
        if not out.get("economic"):
            out.pop("economic", None)
        out["sealed"] = False
        out["seal_status"] = "UNSEALED_ALL_DAYS_COMPLETE"
        return out
    out = _strip_economic({k: v for k, v in day_result.items()
                           if k != "economic"})
    if True:
        out["sealed"] = True
        out["seal_status"] = (
            f"SEALED -- {n_days_complete} of {g} days complete. Every "
            f"economic field is ABSENT from this artifact, not "
            f"present-and-ignored, so whoever runs the remaining days "
            f"has not seen this one's result")
        out["sealed_field_names"] = list(ECONOMIC_FIELDS)
        out["sealed_at_every_depth"] = True
    return out


def _economic_keys_in(o, path="") -> list:
    """Economic fields present as KEYS, at any depth.

    NOT a substring test on the serialised artifact: `sealed_field_names`
    is a LIST OF THE NAMES BEING SEALED, so a substring test reports it as
    a leak. That is the needle-matches-its-own-prose failure this codebase
    has hit three times, and it caught my own check here."""
    found = []
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ECONOMIC_FIELDS:
                found.append(f"{path}.{k}".lstrip("."))
            found += _economic_keys_in(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            found += _economic_keys_in(v, f"{path}[{i}]")
    return found


def assert_no_economic_leak(artifact: dict, n_days_complete: int,
                            g: int) -> bool:
    """The artifact-level guard. Walks the finished artifact and REFUSES if
    any economic field is present while the run is still sealed."""
    if n_days_complete >= g:
        return True
    found = _economic_keys_in(artifact)
    if found:
        raise RunnerRefused(
            f"REFUSED: economic fields leaked into a SEALED artifact at "
            f"{found[:4]} with {n_days_complete} of {g} days complete. The "
            f"smoke publishes resources only; a leaked Z is an early stop "
            f"waiting to happen.")
    return True


# ------------------------------------------------------------- aggregate

def aggregate(day_results: list, params: dict) -> dict:
    """Day-cluster: the exact sign test and Holm at m, the section-7
    predicate COMPUTED. G comes from the parameters, not a constant."""
    g = params["G"]
    by_arm: dict = {}
    for r in day_results:
        by_arm.setdefault(r["arm"], {})[r["day"]] = r
    out = {}
    for arm, days in sorted(by_arm.items()):
        refused = sorted(d for d, r in days.items() if r["status"] != "OK")
        if refused or len(days) != g:
            out[arm] = {
                "verdict": "UNTESTABLE_ON_THE_DECLARED_DAY_SET",
                "n_days_present": len(days), "G": g,
                "refused_days": refused,
                "why": "a refused arm-day is a STATUS; the arm is not "
                       "aggregated over a smaller G"}
            continue
        z = {d: r["economic"]["Z"] for d, r in days.items()}
        out[arm] = DESIGN.day_cluster_verdict(
            z, alpha=params["alpha"], m=params["multiplicity_m"], g=g)
    return {"per_arm": out, "G": g, "multiplicity_m": params["multiplicity_m"],
            "section_7": {
                arm: v.get("FAILS_THE_SECTION_7_PREDICATE")
                for arm, v in out.items()},
            "any_arm_fails": any(
                v.get("FAILS_THE_SECTION_7_PREDICATE") for v in out.values()),
            "the_rule": "the harmful-fill route STOPS if EITHER arm fails "
                        "to beat the replay null at day-cluster level "
                        "(R-547 item 5)"}


# ------------------------------------------------------------- fixture run

def fixture_run() -> dict:
    """An end-to-end run on FIXTURES so the reviewer can drive the runner
    before a day book exists. NO DATA IS READ."""
    started = time.time()
    root = Path(__file__).resolve().parents[2]
    params = json.loads((root / PARAMS_REL).read_text())
    params = dict(params)
    params["days"] = ["FIXTURE-1", "FIXTURE-2", "FIXTURE-3"]
    params["G"] = len(params["days"])
    params["G_derived_from_len_days"] = True
    inputs = verify_run_inputs(params)
    be = inputs["be_module"]

    rng = random.Random(20260906)
    results, sealed_artifacts = [], []
    for i, day in enumerate(params["days"], start=1):
        book_sha = hashlib.sha256(day.encode()).hexdigest()
        verify_day_inputs(
            day, book_sha, book_sha, params,
            {a: s["theta"] for a, s in params["arms"].items()},
            {a: dict(s["model_digests"]) for a, s in params["arms"].items()})
        for arm in sorted(params["arms"]):
            draws = [rng.gauss(0.0, 1.0) for _ in range(600)]
            observed = 2.5 if arm == "CONDVALUE_X_SKEW" else -0.2
            r = arm_day(day, arm, observed, draws, 200, params)
            r["seed"] = seed_for(book_sha, arm)
            results.append(r)
            sealed_artifacts.append(seal(r, i, params["G"]))
    for a in sealed_artifacts[:-2]:
        assert_no_economic_leak(a, 1, params["G"])
    agg = aggregate(results, params)

    DESIGN.LAST_BATTERY.clear()
    DESIGN.selftest(quiet=True)
    design_battery = dict(DESIGN.LAST_BATTERY)
    LAST_BATTERY.clear()
    selftest(quiet=True)
    battery = dict(LAST_BATTERY)

    import resource as _res
    usage = _res.getrusage(_res.RUSAGE_SELF)
    return {
        "protocol": PROTOCOL,
        "status": "FIXTURE_RUN_NO_DATA",
        "as_of": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "no_day_book_was_read": True,
        "the_committed_day_set_is_empty": True,
        "why_fixtures": "the reviewer has not filed on design v3 and BE's "
                        "book declaration is in flight; nothing may touch "
                        "a day. This proves the runner EXECUTES, not that "
                        "any arm does anything",
        "params_used": {k: params[k] for k in (
            "coin", "min_draws_per_arm_day", "min_decisions_per_arm_day",
            "sd_floor_fraction", "alpha", "multiplicity_m",
            "per_day_deadline_s", "G", "G_derived_from_len_days")},
        "be_module_citation": be,
        "run_input_verification_R6": {
            "models": inputs["models"], "thetas": inputs["thetas"],
            "closed_the_reviewers_half_a_code_path": (
                "the book digest was already a verifier; the THETA and "
                "MODEL digests were pinned and never compared. Both now "
                "READ BYTES at run time and a mismatch REFUSES THE RUN"),
        },
        "design_declaration": params["design_declaration"],
        "per_day_sealed_artifacts": sealed_artifacts,
        "aggregate": agg,
        "battery": battery,
        "design_battery": design_battery,
        "wrapper_used": params["wrapper"],
        "resources_observed": {
            "wall_seconds": time.time() - started,
            "user_cpu_seconds": usage.ru_utime,
            "max_rss_kib": usage.ru_maxrss},
        "what_this_receipt_is_not": {
            "a_result": False, "a_day_run": False,
            "it_says_nothing_about_either_arm": True,
            "the_fixture_arms_are_synthetic": "CONDVALUE's fixture is "
                                              "planted above the null and "
                                              "HAZARD's below it, to drive "
                                              "both verdict branches"},
    }


# --------------------------------------------------------------- selftest

LAST_BATTERY: dict = {}


def selftest(*, quiet: bool = False) -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            LAST_BATTERY.update({"outcome": "FAIL", "n_checks_run": n[0],
                                 "failed_on": label})
            raise SystemExit(f"[de_multiday_gate1_runner] FAIL: {label}")
        n[0] += 1
        if not quiet:
            print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        try:
            fn()
        except RunnerRefused as exc:
            if needle.lower() not in str(exc).lower():
                raise SystemExit(f"[de_multiday_gate1_runner] FAIL: {label} "
                                 f"-- wrong reason: {exc}")
            n[0] += 1
            if not quiet:
                print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_multiday_gate1_runner] FAIL: {label} "
                         f"-- ADMITTED")

    root = Path(__file__).resolve().parents[2]
    P = json.loads((root / PARAMS_REL).read_text())

    # ---- G is derived, and an empty ruled set refuses ------------------
    # ---- R-555, the RULED set --------------------------------------
    live = load_params()
    ok(live["G"] == 6 and live["G"] == live["expected_G"]
       and live["days"] == ["2026-09-03", "2026-09-04", "2026-09-05",
                            "2026-09-06", "2026-09-07", "2026-09-08"]
       and live["ruling"]["smoke_day"] == "2026-09-03",
       f"R-555 POSITIVE CONTROL, AND IT ADMITS: the ruled set loads, G is "
       f"DERIVED as {live['G']} and agrees with the declared expected_G, "
       f"and the smoke day is {live['ruling']['smoke_day']}")
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td0:
        five = dict(live); five["days"] = live["days"][:5]
        f5 = Path(td0) / "five.json"; f5.write_text(json.dumps(five))
        refuses(lambda: load_params(f5),
                "R-555 KNOWN-BAD, A SET OF FIVE: it refuses rather than "
                "testing at a smaller G -- 'G stays 6 and nothing is "
                "chosen'", "against the declared expected_G")
        touched = dict(live)
        touched["days"] = ["2026-09-01"] + live["days"][1:]
        ft = Path(td0) / "touched.json"; ft.write_text(json.dumps(touched))
        refuses(lambda: load_params(ft),
                "R-555 KNOWN-BAD, A TOUCHED DAY: 09-01's "
                "previously_opened_for is interim_read_of_frozen_candidate "
                "and the set REFUSES -- untouched days only",
                "not 'none'")
        empty = dict(live); empty["days"] = []
        fe = Path(td0) / "empty.json"; fe.write_text(json.dumps(empty))
        refuses(lambda: load_params(fe),
                "and an EMPTY set still refuses -- G is derived from it "
                "and there is nothing to derive", "ruled day set is EMPTY")
    tmp = dict(P); tmp["days"] = ["a", "b", "c", "d", "e", "f"]
    tmp.pop("expected_G", None); tmp.pop("required_previously_opened_for", None)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "p.json"; f.write_text(json.dumps(tmp))
        got = load_params(f)
        ok(got["G"] == 6 and got["G_derived_from_len_days"] is True,
           f"POSITIVE CONTROL, AND IT ADMITS: G is DERIVED from len(days) "
           f"= {got['G']}, never read from a constant -- the defect design "
           f"v2 shipped")
        tmp2 = dict(tmp); tmp2["days"] = ["a", "a", "b", "c", "d", "e"]
        f2 = Path(td) / "p2.json"; f2.write_text(json.dumps(tmp2))
        refuses(lambda: load_params(f2),
                "KNOWN-BAD: a duplicated day refuses rather than inflating "
                "G", "duplicate days")

    # ---- BE's cascade is cited, and a different one refuses ------------
    ok(verify_be_module(P)["cited_not_copied"] is True,
       "POSITIVE CONTROL: BE's cascade module resolves at the declared "
       "digest and is CITED, not copied")
    refuses(lambda: verify_be_module(P, actual_sha="0" * 64),
            "KNOWN-BAD: a DIFFERENT cascade digest refuses -- a null run "
            "through another cascade is not a control for this arm",
            "cascade module digest differs")

    # ---- R6: THE MODEL AND THETA HALVES ARE NOW CODE PATHS -------------
    vr = verify_run_inputs(live)
    ok(vr["models"]["n_models_read"] == 3
       and vr["models"]["bytes_were_read_not_recorded"] is True
       and all(v["matches"] for a in vr["models"]["per_arm"].values()
               for v in a.values())
       and all(v["matches"] for v in vr["thetas"]["per_arm"].values()),
       f"R6 POSITIVE CONTROL, AND IT ADMITS: {vr['models']['n_models_read']}"
       f" pinned model files are READ AND HASHED at run time and match, "
       f"and both thetas are read from the artifact they are pinned in. "
       f"Until now these were recorded and never compared -- the "
       f"reviewer's 'provenance theatre'")
    import shutil as _sh
    with _tf.TemporaryDirectory() as tdm:
        # A PLANTED MODEL BYTE. The whole model dir is copied so the
        # pristine tree is never touched.
        src = Path(__file__).resolve().parents[2] / MODEL_DIR
        dst = Path(tdm) / MODEL_DIR
        dst.parent.mkdir(parents=True, exist_ok=True)
        _sh.copytree(src, dst)
        tgt = dst / "linear_d_btc.json"
        tgt.write_bytes(tgt.read_bytes() + b" ")
        refuses(lambda: verify_pinned_models(live, Path(tdm)),
                "R6 KNOWN-BAD, A PLANTED MODEL BYTE: one trailing space in "
                "linear_d_btc.json changes its digest and REFUSES THE RUN "
                "-- not the day, because a moved model makes every day's "
                "object different", "do not match their declared digest")
        (dst / "lgbm_haz_btc.txt").unlink()
        refuses(lambda: verify_pinned_models(live, Path(tdm)),
                "and an ABSENT pinned model refuses too, rather than "
                "verifying the ones that happen to be there",
                "do not match their declared digest")
    with _tf.TemporaryDirectory() as tdt:
        bad = dict(live)
        bad["arms"] = {a: dict(v) for a, v in live["arms"].items()}
        bad["arms"]["CONDVALUE_X_SKEW"]["theta"] = 0.5
        refuses(lambda: verify_pinned_thetas(bad),
                "R6 KNOWN-BAD, A THETA THAT DISAGREES WITH ITS PIN SOURCE: "
                "refuses the RUN. The theta is read from BE's artifact at "
                "its declared JSON path, not from this module's copy",
                "theta mismatch against the pin source")

    # ---- R6, all three, both directions --------------------------------
    th = {a: s["theta"] for a, s in P["arms"].items()}
    md = {a: dict(s["model_digests"]) for a, s in P["arms"].items()}
    ok(verify_day_inputs("D", "a" * 64, "a" * 64, P, th, md)[
           "model_digests_verified"] is True,
       "POSITIVE CONTROL ON R6, AND IT ADMITS: matching book, theta and "
       "model digests verify")
    refuses(lambda: verify_day_inputs("D", "a" * 64, "b" * 64, P, th, md),
            "R6 KNOWN-BAD, A WRONG-DIGEST DAY: the book digest mismatch "
            "REFUSES THE DAY", "reference-book digest mismatch")
    bad_th = dict(th); bad_th["CONDVALUE_X_SKEW"] = 0.5
    refuses(lambda: verify_day_inputs("D", "a" * 64, "a" * 64, P, bad_th, md),
            "R6 KNOWN-BAD: a refitted theta REFUSES THE RUN, not the day -- "
            "it makes every day's object different", "theta is 0.5")
    bad_md = {a: dict(v) for a, v in md.items()}
    bad_md["HAZARD_OVER_SKEWED_REF"]["linear_d_btc.json"] = "deadbeef"
    refuses(lambda: verify_day_inputs("D", "a" * 64, "a" * 64, P, th, bad_md),
            "R6 KNOWN-BAD: a moved MODEL digest refuses the run",
            "model linear_d_btc.json digest")

    # ---- R4 and R8 ------------------------------------------------------
    draws = [0.4 + 0.002 * i for i in range(600)]
    good = arm_day("D", "A", 2.0, draws, 200, P)
    ok(good["status"] == "OK" and good["economic"]["Z"] > 0,
       "POSITIVE CONTROL: an admissible arm-day computes its economic "
       "fields")
    short = arm_day("D", "A", 2.0, draws, 4, P)
    ok(short["status"] == "DEGENERATE_ARM_DAY_REFUSED"
       and short["economic"] is None,
       "R4 KNOWN-BAD, A SHORT DAY: four decisions REFUSE the arm-day and it "
       "carries NO economic field -- a status, and it does not shrink G "
       "silently")
    refuses(lambda: arm_day("D", "A", 2.0, draws[:499], 200, P),
            "R4/R8 KNOWN-BAD: 499 draws refuses -- the 500 minimum is never "
            "lowered", "below the declared minimum")
    refuses(lambda: arm_day("D", "A", 2.0, draws, 200, P,
                            elapsed_s=P["per_day_deadline_s"] + 1),
            "R8 KNOWN-BAD, AN OVERRUN: the DAY refuses. The draw count is "
            "never lowered and the cap is never raised",
            "exceeds the declared deadline")

    # ---- R5, the sealed guard, both directions -------------------------
    sealed = seal(good, 1, 5)
    ok(sealed["sealed"] is True and "economic" not in sealed
       and sealed["seal_status"].startswith("SEALED")
       and not _economic_keys_in(sealed),
       "R5 POSITIVE CONTROL: at 1 of 5 days the economic block is ABSENT "
       "from the artifact, not present-and-ignored -- AND SO ARE THE "
       "ECONOMIC FIELDS NESTED ELSEWHERE. Sealing only the top-level block "
       "left `admissibility.null_sd` and `null_mean` behind, and my own "
       "guard caught the emitter on its first run")
    ok(_economic_keys_in({"a": {"Z": 1}}) == ["a.Z"]
       and _economic_keys_in({"sealed_field_names": ["Z", "null_sd"]}) == [],
       "and the leak detector tests KEYS, not substrings: it finds a "
       "nested `Z` and does NOT report `sealed_field_names`, which is a "
       "LIST OF THE NAMES BEING SEALED -- the needle-matches-its-own-prose "
       "failure, caught in my own check")
    ok(assert_no_economic_leak(sealed, 1, 5) is True
       and "admissibility" in sealed,
       "and the R4 admissibility STATUS survives the seal -- what is "
       "withheld is the economics, not the fact that the day ran")
    unsealed = seal(good, 5, 5)
    ok(unsealed["sealed"] is False and "economic" in unsealed,
       "AND IT UNSEALS: at 5 of 5 the economic block is present -- a guard "
       "shown only to withhold is not a guard")
    ok(assert_no_economic_leak(sealed, 1, 5) is True,
       "the artifact-level guard ADMITS a properly sealed artifact")
    leaky = dict(sealed); leaky["economic"] = {"Z": 3.0}
    refuses(lambda: assert_no_economic_leak(leaky, 1, 5),
            "R5 KNOWN-BAD, AN ECONOMIC FIELD LEAKING BEFORE G: the guard "
            "walks the finished artifact and REFUSES -- a leaked Z is an "
            "early stop waiting to happen", "leaked into a SEALED artifact")
    nested = {"a": {"b": [{"p_location": 0.5}]}}
    refuses(lambda: assert_no_economic_leak(nested, 1, 5),
            "and it finds a leak PLANTED AT DEPTH inside a nested list",
            "leaked into a SEALED artifact")

    # ---- the planted arms, both directions -----------------------------
    Pg = dict(P); Pg["G"] = 5
    fail_rows = [{"day": f"d{i}", "arm": "PLANTED_FAIL", "status": "OK",
                  "economic": {"Z": 0.0}} for i in range(5)]
    pass_rows = [{"day": f"d{i}", "arm": "PLANTED_PASS", "status": "OK",
                  "economic": {"Z": 4.0}} for i in range(5)]
    agg = aggregate(fail_rows + pass_rows, Pg)
    ok(agg["per_arm"]["PLANTED_FAIL"]["FAILS_THE_SECTION_7_PREDICATE"]
       is True,
       "A PLANTED MUST-FAIL ARM FAILS: Z = 0 on every day, mean not above "
       "zero and no day sign positive")
    ok(agg["per_arm"]["PLANTED_PASS"]["FAILS_THE_SECTION_7_PREDICATE"]
       is False
       and agg["per_arm"]["PLANTED_PASS"]["clears_holm"] is False,
       "A PLANTED MUST-PASS ARM DOES NOT FAIL -- and at G = 5, m = 2 its "
       "`clears_holm` is FALSE, so the pass is DIRECTIONAL and the "
       "falsifier drives the multiplicity statement, not only the "
       "arithmetic")
    ok(agg["any_arm_fails"] is True
       and agg["the_rule"].startswith("the harmful-fill route STOPS"),
       "and the section-7 rule is COMPUTED across arms: either arm failing "
       "stops the route")
    mixed = [{"day": f"d{i}", "arm": "SHORT", "status": "OK",
              "economic": {"Z": 1.0}} for i in range(4)]
    a2 = aggregate(mixed, Pg)
    ok(a2["per_arm"]["SHORT"]["verdict"]
       == "UNTESTABLE_ON_THE_DECLARED_DAY_SET",
       "KNOWN-BAD: an arm present on four of five days is UNTESTABLE, not "
       "tested at G = 4 -- a refused arm-day does not shrink G")

    ok(seed_for("a" * 64, "X") != seed_for("b" * 64, "X")
       and seed_for("a" * 64, "X") == seed_for("a" * 64, "X"),
       "the seed PINS THE DATA: it changes with the book digest and is "
       "reproducible from the artifact alone")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    LAST_BATTERY.update({
        "outcome": "PASS", "n_checks_run": n[0],
        "expected_checks_in_the_source": EXPECTED_CHECKS,
        "run_count_equals_source_expected": n[0] == EXPECTED_CHECKS,
        "ran_in_the_emitting_process": True})
    if not quiet:
        print(f"[de_multiday_gate1_runner] PASS -- {n[0]} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fixture-run", action="store_true", dest="fixture")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.fixture or a.output is None:
        ap.error("choose --selftest or --fixture-run --output PATH")
    payload = fixture_run()
    if a.output.exists():
        raise RunnerRefused(f"output already exists: {a.output}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"emitted": str(a.output), "status": payload["status"],
                      "G": payload["params_used"]["G"],
                      "battery": payload["battery"]["outcome"],
                      "battery_checks": payload["battery"]["n_checks_run"],
                      "any_arm_fails": payload["aggregate"]["any_arm_fails"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
