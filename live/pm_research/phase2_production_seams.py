"""Seams 17-21: enter where PRODUCTION enters. R-199 design rule.

AUTHORISATION (R-126, in-file): R-199, user audit #4.

WHY THIS FILE EXISTS SEPARATELY FROM THE FIXTURE. Seams 1-16 assert
CAPABILITIES: a function exists, a signature accepts an argument, a mapping
returns the right kind, a field reaches a receipt. Four user audits have now
found defects that ALL live in WIRING -- an argument never passed, a variable
never bound, a function never called, an import that reads the wrong tree.
Capability assertions cannot see any of those, because each one is true of a
pipeline that never runs.

THE RULE: a seam enters where production enters. These call the REAL
stage_fit / stage_score / builder entry points on SYNTHETIC ARTIFACTS and
assert OUTCOMES -- did it complete, what did it produce, does it refuse.
"""
from __future__ import annotations

import json, os, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FAILURES: list = []


def seam(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")
        FAILURES.append(name)


def _mini_exposure(path: Path, slug_t0: float, n_gen: int, coin: str, day: str):
    rows = []
    for g in range(n_gen):
        for k in range(2):
            rows.append({"slug": f"{coin}-updown-5m-{int(slug_t0)}", "coin": coin,
                         "day": day, "t0": slug_t0, "t_start": g * 3.0 + k * 0.5,
                         "side": "BUY_UP" if g % 2 else "SELL_UP", "gen": g,
                         "status": "OK", "level": 0.5, "resting": 5.0,
                         "qahead": 2.0,
                         "latency": {"50": {"preventable_value_cents": (g % 5) - 2.0,
                                            "preventable_shares": 1.0 if g % 2 else 0.0,
                                            "stale_shares": 0.0}},
                         "any_fill_ahead": bool(g % 2)})
    path.write_text(json.dumps({"rows": rows, "days": [day], "n_windows": 1,
                                "schema": "harmful_exposure_v3_4_fill_scoped_markout"}))
    return rows


def main() -> int:
    import phase2_arms as PA
    import phase2_state_schema_freeze as PIN
    import phase2_declaration as D

    feats = PIN.build_pin()["features_in_order"]

    # ---- SEAM 21: refusal wiring must have CALL SITES --------------------
    import inspect
    fit_src = inspect.getsource(PA.stage_fit)
    sc_src = inspect.getsource(PA.stage_score)
    seam("21a stage_fit CALLS assert_tape_is_v5",
         "assert_tape_is_v5(" in fit_src,
         "defined at :67 with zero call sites -- a refusal nobody invokes")
    seam("21b stage_score CALLS assert_tape_is_v5",
         "assert_tape_is_v5(" in sc_src)
    _sv = PA.DA_VERDICT
    PA.DA_VERDICT = Path("/nonexistent/da_tape_gate_verdict_v5.json")
    try:
        PA.assert_gate_passed()
        seam("21c fitting REFUSES without DA's ALL-PASS verdict", False,
             "an absent verdict was accepted as permission")
    except RuntimeError as e:
        seam("21c fitting REFUSES without DA's ALL-PASS verdict",
             "absence is not permission" in str(e))
    finally:
        PA.DA_VERDICT = _sv

    # ---- SEAM 20: provenance interface + snapshot import isolation -------
    b_src = (HERE / "build_state_tape_v2.py").read_text()
    seam("20a builder reads BUILD_REF from the ENV",
         'environ["BUILD_REF"]' in b_src or 'environ.get("BUILD_REF"' in b_src,
         "builder runs git at runtime instead of being told its ref")
    import ast as _ast20
    _git_calls = []
    for _n in _ast20.walk(_ast20.parse(b_src)):
        if isinstance(_n, _ast20.Constant) and isinstance(_n.value, str) \
           and _n.value == "git":
            _git_calls.append(getattr(_n, "lineno", -1))
    seam("20b builder invokes NO git at runtime (AST, not grep)",
         not _git_calls,
         f"git invoked at lines {_git_calls}; a comment mentioning rev-parse "
         f"is not an invocation and a grep cannot tell them apart")
    seam("20c import root comes from __file__, not a hardcoded path",
         'sys.path.insert(0, "/home/yuqing/ctaNew' not in b_src,
         "a hardcoded main-tree sys.path makes a snapshot import the LIVE tree "
         "-- the snapshot isolates nothing")
    seam("20d a preamble probe asserts modules loaded UNDER the snapshot root",
         "__file__" in b_src and "snapshot" in b_src.lower(),
         "without it, wrong-tree imports are silent")

    # ---- SEAM 17: stage_fit END-TO-END on a synthetic v5 tape ------------
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        frag = tdp / "frag.json"; top = tdp / "top.json"
        fr = _mini_exposure(frag, 1_000_000.0, 12, "btc", "d1")
        sc = _mini_exposure(top, 1_000_500.0, 8, "btc", "d2")
        # eth rows too, so the NON-EMPTY path is exercised on both coins and
        # the fixture is not silently a single-coin test
        fr += _mini_exposure(tdp / "frag_e.json", 1_000_000.0, 10, "eth", "d1")
        sc += _mini_exposure(tdp / "top_e.json", 1_000_500.0, 6, "eth", "d2")
        frag.write_text(json.dumps({"rows": fr, "days": ["d1"], "n_windows": 2,
            "schema": "harmful_exposure_v3_4_fill_scoped_markout"}))
        top.write_text(json.dumps({"rows": sc, "days": ["d2"], "n_windows": 2,
            "schema": "harmful_exposure_v3_4_fill_scoped_markout"}))
        tape = {"protocol": "PHASE2_STATE_TAPE_V5", "features_under": "state",
                "builder_ref": "0" * 40, "builder_tree_dirty_at_build": False,
                "rows": []}
        for split, rows in (("train", fr), ("score", sc)):
            for r in rows:
                tape["rows"].append({**{k: r[k] for k in
                                        ("slug", "coin", "day", "t0", "t_start",
                                         "side", "gen")},
                                     "split": split, "state_status": "OK",
                                     "decision_time": r["t_start"],
                                     "state": {k: 0.25 for k in feats}})
        tp = tdp / "tape.json"; tp.write_text(json.dumps(tape))
        # R-203(3): the fixture's verdict must satisfy the RULED CONTRACT --
        # identity, a real predicate table, and binding to THIS tape by hash,
        # bytes and builder_ref. A bare {"verdict":"PASS"} is exactly what the
        # contract exists to refuse, so a fixture using one tests nothing.
        import hashlib as _h
        _tb = tp.stat().st_size
        _hh = _h.sha256(tp.read_bytes()).hexdigest()
        vd = tdp / "verdict.json"
        vd.write_text(json.dumps({
            "verdict": "da_tape_gate_verdict_v1",
            # the RULED load-bearing set, each ASSERTED and PASSING, plus
            # embargo_respected as the one legitimate N/A. A fixture whose
            # verdict does not carry the real predicate names would pass a
            # consumer that never checks for them.
            "predicates": [
                {"predicate": "gap_count_matches_expected", "pass": True,
                 "applicable": True},
                {"predicate": "provenance_matches_expected", "pass": True,
                 "applicable": True},
                {"predicate": "dataset_non_empty", "pass": True,
                 "applicable": True},
                {"predicate": "no_rows_skipped_by_builder", "pass": True,
                 "applicable": True},
                {"predicate": "absorption_within_bound", "pass": True,
                 "applicable": True},
                {"predicate": "embargo_respected", "pass": False,
                 "applicable": False, "detail": "ENFORCED-DOWNSTREAM"}],
            "tape_path": str(tp), "tape_bytes": _tb,
            "tape_sha256_prefix": _hh[:16],
            "builder_ref": "0" * 40,
            "synthetic_note": "contract-conforming fixture verdict; the real "
                              "gate emits this shape on the real tape"}))
        # R-203(4): PA.OUT MUST be sandboxed. The seam run wrote the REAL
        # phase2_three_arm_v1.json -- a test overwriting an evidentiary
        # artifact, the same class as filtering a rejected build in place.
        saved = (PA.TAPE_PATH, PA.FRAGMENT, PA.TOPUP, PA.FITDIR,
                 PA.DA_VERDICT, PA.OUT)
        PA.OUT = tdp / "phase2_three_arm_SANDBOX.json"
        PA.TAPE_PATH, PA.FRAGMENT, PA.TOPUP = tp, frag, top
        PA.FITDIR = tdp / "fits"
        PA.DA_VERDICT = vd
        import harmful_hazard_model as _hm
        _saved_fn = (_hm.features, _hm.fine_feats, _hm.window_streams,
                     _hm.fi._archive_paths, _hm.fi.token_map)
        _slugs = {r["slug"] for r in fr} | {r["slug"] for r in sc}
        # the stub must match the FROZEN candidate's PM+fine width, or the
        # width guard fires on the fixture rather than on a real mismatch
        _fzw = json.loads(PA.FROZEN.read_text())["fits"]["btc"]["norm_mu"]
        _npm = len(_fzw) - 1
        _hm.features = lambda *a, **k: [0.3] * _npm
        _hm.fine_feats = lambda *a, **k: [0.1]
        _hm.window_streams = lambda *a, **k: object()
        _hm.fi._archive_paths = lambda *a, **k: {x: Path("/dev/null") for x in _slugs}
        _hm.fi.token_map = lambda *a, **k: {x: ("u", "d") for x in _slugs}
        try:
            PA.stage_fit()
            seam("17a stage_fit COMPLETES end-to-end on a synthetic v5 tape", True)
            for coin in ("btc", "eth"):
                seam(f"17b arm-D artifact written for {coin}",
                     (PA.FITDIR / f"linear_d_{coin}.json").exists())
        except Exception as e:
            seam("17a stage_fit COMPLETES end-to-end on a synthetic v5 tape",
                 False, f"{type(e).__name__}: {e}")
            seam("17b arm-D artifact written for btc", False, "fit did not complete")
        # ---- SEAM 18: stage_score END-TO-END, all four arms --------------
        try:
            PA.stage_score()
            seam("18a stage_score COMPLETES with ALL FOUR arms", True)
        except Exception as e:
            seam("18a stage_score COMPLETES with ALL FOUR arms", False,
                 f"{type(e).__name__}: {e}")
        finally:
            (_hm.features, _hm.fine_feats, _hm.window_streams,
             _hm.fi._archive_paths, _hm.fi.token_map) = _saved_fn
            (PA.TAPE_PATH, PA.FRAGMENT, PA.TOPUP, PA.FITDIR,
             PA.DA_VERDICT, PA.OUT) = saved

    # ---- SEAM 19: causal thresholds actually reach the evaluator ---------
    seam("19a stage_score PASSES theta_frozen to evaluate_policy",
         "theta_frozen=" in sc_src,
         "thresholds are frozen, persisted, and never passed -- the evaluator "
         "still computes theta retrospectively")
    seam("19b thresholds are ACTION/generation maxima, not row quantiles",
         "gen" in inspect.getsource(PA.freeze_thresholds).lower(),
         "a row-quantile cutoff is not comparable to a per-generation max, so "
         "the frozen theta selects the wrong count")
    seam("19c arms A and C get their OWN training thresholds",
         "causal_thresholds" in fit_src and fit_src.count("causal_thresholds") >= 3,
         "only B and D persist thresholds; A and C have none")

    # ---- SEAM 22: a snapshot must not relocate the DATA ROOT -------------
    # tape6 died with KeyError on every slug because flow_intensity derives
    # REPO from its own __file__ (:44-45), so a CODE snapshot silently moved
    # the DATA root with it and token_map() returned 0 entries. R-197 assumed
    # "data paths are absolute in the builder" -- true of the builder, false of
    # a module it calls. The builder must PIN the data root explicitly.
    b_src2 = (HERE / "build_state_tape_v2.py").read_text()
    seam("22a builder PINS the data root against snapshot relocation",
         "PM_DATA_ROOT" in b_src2 or "fi.PM =" in b_src2 or "fi.REPO =" in b_src2,
         "a code snapshot relocates any module that derives data from __file__")
    import harmful_hazard_model as _hm22
    seam("22b the pinned root points at the REAL data tree",
         str(_hm22.fi.PM).startswith("/home/yuqing/ctaNew/data"),
         f"fi.PM = {_hm22.fi.PM}")
    # BEHAVIOURAL: the data must LOAD, not merely look addressable. The first
    # data-root fix rebound PM and left RAW/GAPS/MARKETS derived from the old
    # value, so every path looked right and token_map() was still empty.
    # BEHAVIOURAL: run the real preflight and require it to report a
    # row-path probe over KNOWN population slugs. Greping for a message string
    # went RED the moment the message was reworded -- seventh recurrence of
    # source-text matching misreporting, so this asserts the OUTCOME.
    import io as _io22, contextlib as _cl22
    import build_state_tape_v2 as _B22
    _buf = _io22.StringIO()
    try:
        with _cl22.redirect_stdout(_buf):
            _B22.pin_data_root()
        _out = _buf.getvalue()
        seam("22c the preflight probes KNOWN slugs through the row path",
             "row_path_probe" in _out and "/" in _out.split("row_path_probe")[1][:12],
             f"preflight said: {_out.strip()[:120]}")
    except SystemExit as _e:
        seam("22c the preflight probes KNOWN slugs through the row path",
             False, f"preflight REFUSED: {_e}")

    # ---- SEAM 23: an unmappable slug is a STATUS, never a KeyError --------
    b_main = b_src2[b_src2.find("def main"):]
    seam("23a builder handles a slug with no token mapping as a STATUS",
         "NO_TOKEN_MAP" in b_src2,
         "tokens[slug] raised KeyError and killed the build (rule 4: an "
         "exclusion is a counted status, never a crash and never a silent skip)")
    seam("23b it is COUNTED, not silently skipped",
         "NO_TOKEN_MAP" in b_src2 and "status_counts" in b_src2)

    # ---- SEAM 24 (R-202): a status must never absorb a TOTAL failure -----
    b24 = (HERE / "build_state_tape_v2.py").read_text()
    seam("24a zero-row builds are REFUSED, never written",
         "produced ZERO rows" in b24,
         "tape6b wrote a 0-row artifact reporting 'embargo CERTIFIED'")
    seam("24b any status above 1% of input rows REFUSES",
         "absorption bound" in b24 and "0.01" in b24,
         "a status absorbing 1,764,206 of 1,764,206 rows exited 0")
    seam("24c the preflight probes the ROW PATH, not just the load",
         "row_path_probe" in b24,
         "loading a map is not the lookup the row path performs")
    seam("24d exclusion statuses NAME their cause",
         "NO_ARCHIVE_PATH" in b24 and "NO_TOKEN_MAP" in b24,
         "one status covering two causes misdirected the diagnosis")
    seam("24e every builder INPUT is verified, not just one",
         "archive_paths" in b24 and "gaps_by_slug" in b24 and "DAYS" in b24,
         "token_map alone passed while archive_paths was empty")

    # ================= SEAM 25 (R-203): the user's six probes ============
    import harmful_action_eval as ae
    b25 = (HERE / "build_state_tape_v2.py").read_text()
    a25 = inspect.getsource(PA)

    # 25a/b: the bound applies to PRE-EMISSION SKIPS only
    seam("25a skips are tallied SEPARATELY from emitted statuses",
         "skip_counts" in b25 and "pre_emission_skip_counts" in b25,
         "one tally meant PRE_WINDOW at 3.85% would refuse a VALID population")
    seam("25b the 1% bound iterates skips, not statuses",
         "for _st, _n in sorted(skip_counts.items())" in b25,
         "iterating status_counts refuses the good build at completion")

    # 25c: a fabricated verdict must be REFUSED
    import tempfile as _t25, json as _j25
    _d25 = Path(_t25.mkdtemp()); _v25 = _d25 / "v.json"
    _v25.write_text(_j25.dumps({"verdict": "PASS"}))
    _sv25 = PA.DA_VERDICT; PA.DA_VERDICT = _v25
    try:
        PA.assert_gate_passed()
        seam("25c a fabricated {'verdict':'PASS'} is REFUSED", False,
             "any string can be written into a file")
    except RuntimeError as _e25:
        seam("25c a fabricated {'verdict':'PASS'} is REFUSED",
             "ruled contract" in str(_e25))
    finally:
        PA.DA_VERDICT = _sv25
    seam("25d the verdict is RECOMPUTED from the predicate table",
         "recomputed all_pass" in a25 or "recomputed" in a25,
         "trusting a summary field is trusting a string")
    seam("25e the verdict is BOUND to the tape by hash, bytes and ref",
         "tape_sha256_prefix" in a25 and "tape_bytes" in a25
         and "builder_ref" in a25)

    # 25f: zero cancellations is valid
    ev_src25 = inspect.getsource(ae.evaluate_policy)
    seam("25f zero cancellations is VALID (no order[:1] force)",
         "cancelled = order[:1]" not in ev_src25 and
         "cancelled NOTHING" in ev_src25,
         "forcing one cancel invents a decision the threshold declined")

    # 25g: multiplicity COMPUTED, not a literal
    seam("25g multiplicity is COMPUTED from the declaration",
         "D.MULTIPLICITY_BEFORE + len(D.WEIGHTED_ARMS_V2)" in a25,
         "a literal beside its own declaration has contradicted it before")

    # 25h: each arm carries ITS OWN thresholds
    seam("25h the receipt carries each arm's OWN thresholds",
         '"causal_thresholds": thr' in a25 and "threshold_source" in a25,
         "the receipt carried B's thresholds for every arm")
    seam("25i D's thresholds come from GENERATION maxima",
         a25.count("gen_keys=_gk") >= 4,
         "a row-quantile cutoff is not comparable to a per-generation max")

    # 25j: value diagnostics condition on the hazard-positive population
    hd25 = inspect.getsource(PA.head_diagnostics)
    seam("25j conditional-value diagnostics condition on hazard-positive rows",
         "if yy" in hd25,
         "scoring a conditional head over rows with no fill measures it on a "
         "population it never claims to describe")
    seam("25k Brier asserts length equality",
         "len(p_haz) == n" in hd25,
         "zip() truncates silently, so a short p_haz gave a confident Brier "
         "over a prefix")

    # ---- SEAM 26 (R-204): all arms in the SAME evaluation mode ----------
    a26 = inspect.getsource(PA.stage_score)
    seam("26a the receipt asserts ALL arms share one evaluation mode",
         "ALL_ARMS_SAME_MODE" in a26,
         "an arm left on RETROSPECTIVE_TOPK while others are causal is not "
         "comparable, and its numbers still look normal alone")
    seam("26b a mode split REFUSES rather than reporting",
         "not a comparison" in a26)

    # ---- SEAM 27 (R-207): N/A-VACUITY -----------------------------------
    # DA's bypass: a gate run with NO expectations marks every predicate
    # not-applicable, writes all_pass:true, and a consumer that merely EXCLUDES
    # N/A accepts it. Excluding N/A from the pass computation was right;
    # allowing a LOAD-BEARING predicate to BE N/A was not. The absence of a
    # check is not the passing of a check.
    import hashlib as _h27, tempfile as _t27, json as _j27
    _d27 = Path(_t27.mkdtemp()); _tp27 = _d27 / "t.json"
    _tp27.write_text(_j27.dumps({"protocol": "PHASE2_STATE_TAPE_V5",
                                 "rows": [{"a": 1}]}))
    _hh27 = _h27.sha256(_tp27.read_bytes()).hexdigest()
    _LB27 = ("gap_count_matches_expected", "provenance_matches_expected",
             "dataset_non_empty", "no_rows_skipped_by_builder",
             "absorption_within_bound")

    def _verdict27(preds):
        _v = _d27 / "v.json"
        _v.write_text(_j27.dumps({
            "verdict": "da_tape_gate_verdict_v1", "all_pass": True,
            "predicates": preds, "tape_path": str(_tp27),
            "tape_bytes": _tp27.stat().st_size,
            "tape_sha256_prefix": _hh27[:16], "tape_header_pins": {}}))
        return _v

    def _accepts27(preds):
        _sv = PA.DA_VERDICT; PA.DA_VERDICT = _verdict27(preds)
        try:
            PA.assert_gate_passed(); return True
        except RuntimeError:
            return False
        finally:
            PA.DA_VERDICT = _sv

    seam("27a an ALL-N/A verdict claiming all_pass is REFUSED",
         not _accepts27([{"predicate": k, "pass": False, "applicable": False}
                         for k in _LB27]))
    seam("27b one trivial pass with load-bearing all N/A is REFUSED",
         not _accepts27([{"predicate": "trivial", "pass": True, "applicable": True}]
                        + [{"predicate": k, "pass": False, "applicable": False}
                           for k in _LB27]),
         "the subtler bypass: a non-empty applicable set that checks nothing "
         "load-bearing")
    seam("27c a MISSING load-bearing predicate is REFUSED",
         not _accepts27([{"predicate": k, "pass": True, "applicable": True}
                         for k in _LB27[:-1]]))
    seam("27d a FAILING load-bearing predicate is REFUSED",
         not _accepts27([{"predicate": k, "pass": (k != "dataset_non_empty"),
                          "applicable": True} for k in _LB27]))
    seam("27e a non-whitelisted N/A predicate is REFUSED",
         not _accepts27([{"predicate": k, "pass": True, "applicable": True}
                         for k in _LB27]
                        + [{"predicate": "schema_family_matches",
                            "pass": False, "applicable": False}]),
         "N/A is a claim that a predicate CANNOT apply, not a way to skip one")
    seam("27f KNOWN-GOOD: all load-bearing asserted+passing, embargo N/A, ACCEPTED",
         _accepts27([{"predicate": k, "pass": True, "applicable": True}
                     for k in _LB27]
                    + [{"predicate": "embargo_respected", "pass": False,
                        "applicable": False}]),
         "the legitimate shape must still pass, or the fix is just a wall")

    # ---- SEAM 28 (R-209): partial theta + partial fit -------------------
    # 28a: a PARTIAL theta_frozen must REFUSE. The old per-budget guard let a
    # missing key fall back to retrospective FOR THAT BUDGET while the arm
    # reported causal -- within-arm mixing the same-mode check cannot see,
    # because it compares arms rather than budgets.
    _rr28 = [{"slug": "s", "side": "BUY_UP", "gen": i, "t_start": float(i),
              "t0": 1000.0, "any_fill_ahead": True,
              "latency": {"50": {"preventable_value_cents": 1.0,
                                 "preventable_shares": 1.0,
                                 "stale_shares": 0.0}}} for i in range(6)]
    try:
        ae.evaluate_policy(_rr28, [0.5] * 6, latency_ms=50,
                           budgets=(0.05, 0.10), n_random=200,
                           theta_frozen={"5%": 0.4})
        seam("28a a PARTIAL theta_frozen is REFUSED", False,
             "the missing budget fell back to retrospective")
    except ValueError as _e28:
        seam("28a a PARTIAL theta_frozen is REFUSED",
             "lacks budget key" in str(_e28))
    _full28 = ae.evaluate_policy(_rr28, [0.5] * 6, latency_ms=50,
                                 budgets=(0.5,), n_random=200,
                                 theta_frozen={"50%": 0.4})
    seam("28b each BUDGET carries its own mode stamp",
         _full28["budgets"]["50%"].get("threshold_mode")
         == "CAUSAL_FROZEN_FROM_TRAIN",
         "one stamp per arm hides within-arm mixing")

    # 28c: a killed partial fit (no manifest) must REFUSE scoring
    _d28 = Path(_t25.mkdtemp()); _fd28 = _d28 / "fits"; _fd28.mkdir()
    (_fd28 / "linear_btc.json").write_text("{}")      # stale partial artifact
    _svf = PA.FITDIR; PA.FITDIR = _fd28
    try:
        PA.assert_fit_complete_and_matching()
        seam("28c a fit with NO completion manifest is REFUSED", False,
             "a killed partial fit looks identical to a finished one")
    except RuntimeError as _e28b:
        seam("28c a fit with NO completion manifest is REFUSED",
             "killed partial" in str(_e28b))
    finally:
        PA.FITDIR = _svf
    seam("28d stage_score REQUIRES the fit manifest",
         "assert_fit_complete_and_matching()" in inspect.getsource(PA.stage_score))
    seam("28e the fit promotes ATOMICALLY from a run dir",
         "os2.replace(str(_run)" in inspect.getsource(PA.stage_fit) or
         "_os2.replace" in inspect.getsource(PA.stage_fit),
         "writing in place into a shared dir is the in-place-overwrite class")

    print(f"\n{'PRODUCTION SEAMS GREEN' if not FAILURES else 'PRODUCTION SEAMS RED'}: "
          f"{len(FAILURES)} failing")
    for f in FAILURES:
        print(f"  - {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
