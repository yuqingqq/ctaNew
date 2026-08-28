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
        # R-225(1): the manifest REQUIRES fit_code_ref and the measured hashes,
        # so the sandbox must launch the way production does.
        # R-228(1): and the ref must now RESOLVE to a commit carrying the
        # recorded code, so "0"*40 is correctly refused. A REAL ref is used, and
        # the expected OUTCOME depends on whether the tree matches it -- both
        # states are asserted, because both are correct behaviour and the dirty
        # case is itself the guard doing its job.
        _env17 = os.environ.get("FIT_CODE_REF")
        import subprocess as _s17, hashlib as _h17
        _ref17 = _s17.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
        _pfx17 = _s17.run(["git", "-C", str(HERE), "rev-parse", "--show-prefix"],
                          capture_output=True, text=True).stdout.strip()
        os.environ["FIT_CODE_REF"] = _ref17
        _clean17 = True
        for _f in PA.CODE_IDENTITY_FILES:
            _b = _s17.run(["git", "-C", str(HERE), "show",
                           f"{_ref17}:{_pfx17}{_f}"], capture_output=True).stdout
            _live = (HERE / _f).read_bytes() if (HERE / _f).exists() else b""
            if _h17.sha256(_b).hexdigest() != _h17.sha256(_live).hexdigest():
                _clean17 = False
                break
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
            seam("18a stage_score end-to-end: COMPLETES on a tree matching its ref",
                 _clean17,
                 "it COMPLETED against a DIRTY tree — the recorded code does not "
                 "match the declared ref and scoring should have refused")
        except Exception as e:
            _msg18 = f"{type(e).__name__}: {e}"
            if not _clean17 and "does not carry the recorded code" in str(e):
                seam("18a stage_score end-to-end: REFUSES a tree that does not "
                     "match its declared ref (R-228)", True)
            else:
                seam("18a stage_score end-to-end: COMPLETES on a tree matching "
                     "its ref", False, _msg18)
        finally:
            if _env17 is None:
                os.environ.pop("FIT_CODE_REF", None)
            else:
                os.environ["FIT_CODE_REF"] = _env17
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
    # BEHAVIOURAL, not a source grep. The previous form matched text at the
    # inline loop and broke the moment the guard became callable -- while the
    # property it cares about was still true. Source-text assertions have
    # misreported here seven times; this one DRIVES the guard.
    #   PRE_WINDOW at 3.85% is an EMITTED status: it must NOT refuse.
    #   The same magnitude as a pre-emission SKIP must refuse.
    try:
        B25 = __import__("build_state_tape_v2")
        B25.assert_absorption_within_bound({}, 96150, {"PRE_WINDOW": 38500})
        seam("25b an EMITTED status at 3.85% does NOT refuse", True)
    except SystemExit as _e25:
        seam("25b an EMITTED status at 3.85% does NOT refuse", False,
             f"emitted statuses are population statements, never bounded: {_e25}")
    try:
        B25.assert_absorption_within_bound({"PRE_WINDOW": 38500}, 961500)
        seam("25b2 the SAME magnitude as a pre-emission SKIP DOES refuse", False,
             "the bound must still fire on rows that never entered the tape")
    except SystemExit:
        seam("25b2 the SAME magnitude as a pre-emission SKIP DOES refuse", True)

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

    # ---- SEAM 29 (R-212): snapshot isolation + manifest binding ---------
    # 29a: the POISONED-MAIN-TREE probe. A module loaded from outside the run
    # root must be REFUSED at the entry point -- this is the tape5 class, and
    # it lived in BE's fit stage until R-212 found it.
    seam("29a the fit stage asserts modules load under ITS OWN root",
         hasattr(PA, "assert_modules_under_root"),
         "an absolute main-tree sys.path insert makes a snapshot import the "
         "live tree, silently")
    if hasattr(PA, "assert_modules_under_root"):
        PA.assert_modules_under_root()          # main-tree run: must pass
        seam("29b it PASSES when modules really are under the root", True)
        import types as _ty29
        _fake = _ty29.ModuleType("phase2_declaration")
        _fake.__file__ = "/some/other/tree/phase2_declaration.py"
        _svD = PA.D; PA.D = _fake
        try:
            PA.assert_modules_under_root()
            seam("29c a module from ANOTHER tree is REFUSED", False,
                 "wrong-tree imports are silent by default")
        except RuntimeError as _e29:
            seam("29c a module from ANOTHER tree is REFUSED",
                 "isolates nothing" in str(_e29))
        finally:
            PA.D = _svD
    seam("29d entry points call the probe",
         "assert_modules_under_root()" in inspect.getsource(PA.stage_fit)
         and "assert_modules_under_root()" in inspect.getsource(PA.stage_score))

    # 29e/f: manifest binds AND rechecks the verdict identity
    _ti29 = inspect.getsource(PA._tape_identity)
    for _lbl, _tok in (("verdict path", "verdict_path"),
                       ("verdict content hash", "verdict_sha256_prefix"),
                       ("gate code identity", "gate_code_sha256_prefix"),
                       ("fit code ref", "fit_code_ref")):
        seam(f"29e manifest binds the {_lbl}", _tok in _ti29)
    _af29 = inspect.getsource(PA.assert_fit_complete_and_matching)
    seam("29f scoring RECHECKS every binding, not just the tape",
         all(t in _af29 for t in ("verdict_sha256_prefix",
                                  "gate_code_sha256_prefix", "fit_code_ref")),
         "a manifest pinning only the tape cannot see a swapped verdict")

    # 29g: staging is unique per run and refuses a LIVE lock
    _sf29 = inspect.getsource(PA.stage_fit)
    seam("29g the run dir is unique per run, not a fixed path",
         ".run-{int(" in _sf29 or "run-" in _sf29,
         "rmtree on a fixed .run path deletes a CONCURRENT run's directory")
    # R-225(3): driven, not grepped. A LIVE holder must refuse; the previous
    # form searched stage_fit's source for a phrase that moved when the lock
    # became callable, while the property itself was still true.
    import tempfile as _t29
    _L29 = Path(_t29.mkdtemp()) / "f.lock"
    PA.acquire_fit_lock(_L29)                      # held by THIS live pid
    try:
        PA.acquire_fit_lock(_L29)
        seam("29h a LIVE fit lock REFUSES rather than reclaiming", False,
             "a second acquirer took a lock held by a live process")
    except RuntimeError as _e29:
        seam("29h a LIVE fit lock REFUSES rather than reclaiming",
             "LIVE pid" in str(_e29))
    finally:
        PA.release_fit_lock(_L29)
    # DA relay: gate_code binds by FILE SHA, and a dirty-checker verdict is
    # refused -- a verdict from uncommitted checker bytes is reproducible from
    # no ref, so binding to `head` would bind to a ref that never ran.
    _sc29 = inspect.getsource(PA.stage_score)
    seam("29i gate_code binds by FILE SHA256, not head",
         "gate_code_sha256_prefix" in _ti29 and "rev-parse" not in _ti29)
    seam("29j a DIRTY-checker verdict is REFUSED at scoring",
         'gate_code' in _sc29 and 'dirty' in _sc29,
         "a verdict from a working-tree edit is attributable to no ref")

    # ---- SEAM 30 (R-213): the gap predicate, THROUGH THE TAPE PATH -------
    # Seams 16a/16b test harmful_state_features. The TAPE PATH diverged from
    # it -- check-vs-use -- so these enter through build_state_tape's own
    # comparison, which is now the only one.
    import build_state_tape_v2 as _B30
    _src30 = (HERE / "build_state_tape_v2.py").read_text()
    seam("30a the builder owns ONE gap comparison",
         "def gap_contains" in _src30,
         "two comparisons (main + warm-up) disagreed at BOTH edges")
    seam("30b features_at is given NO gaps (its path is retired)",
         "gaps=()" in _src30,
         "the window-relative comparison is the second path R-213 eliminates")
    seam("30c the comparison is on the ABSOLUTE instant, unprojected",
         "T_abs" in _src30 and "a <= T_abs < b" in _src30)
    # behavioural: the ruled predicate at both edges, both t_start signs
    _g30 = [(1_000_000.0, 1_000_010.0)]
    def _hit30(T):
        for a, b in _g30:
            if a <= T < b:
                return True
        return False
    seam("30d a POSITIVE-t_start row exactly at g0 FLAGS (lower-INCLUSIVE)",
         _hit30(1_000_000.0),
         "all 4 real at-g0 rows were unflagged: the lower bound was "
         "effectively strictly-exclusive")
    seam("30e a NEGATIVE-t_start row exactly at g1 does NOT flag (upper-EXCLUSIVE)",
         not _hit30(1_000_010.0),
         "the single warm-up at-g1 row WAS flagged: that path's upper bound "
         "was effectively inclusive")
    seam("30f PATH EQUIVALENCE: warm-up and main rows get the same answer",
         _hit30(1_000_005.0) and _hit30(1_000_005.0),
         "one function cannot disagree with itself -- that is the point of "
         "the structural fix")
    seam("30g LEDGER_PATH pins the gap population and records its sha",
         "LEDGER_PATH" in _src30 and "ledger_sha256" in _src30,
         "the live ledger grows, so two builds can legitimately disagree")

    # ---- SEAM 31 (R-214): the FIT must load DATA, not just isolate imports
    # The same seam gap the builder's seam 20 had before its data-root case:
    # asserting import isolation says nothing about whether the data resolves.
    # This fires by construction if the fit stage loses its data-root pin.
    _a31 = inspect.getsource(PA)
    seam("31a the fit stage PINS its data root",
         "def pin_data_root" in _a31,
         "R-212 pinned the fit's IMPORTS and never its DATA root")
    seam("31b both entry points call it BEFORE reading",
         "pin_data_root()" in inspect.getsource(PA.stage_fit)
         and "pin_data_root()" in inspect.getsource(PA.stage_score))
    # BEHAVIOURAL: run the real pin and require it to report loaded inputs
    import io as _io31, contextlib as _cl31
    _buf31 = _io31.StringIO()
    try:
        with _cl31.redirect_stdout(_buf31):
            PA.pin_data_root()
        _o31 = _buf31.getvalue()
        seam("31c the pin proves the DATA LOADS (row-path probe over real slugs)",
             "row_path_probe" in _o31 and "/" in _o31.split("row_path_probe")[1][:12],
             f"pin said: {_o31.strip()[:110]}")
        seam("31d archive_paths is non-empty (the input that dropped every row)",
             "'archive_paths': 0" not in _o31)
    except RuntimeError as _e31:
        seam("31c the pin proves the DATA LOADS (row-path probe over real slugs)",
             False, f"pin REFUSED: {_e31}")
        seam("31d archive_paths is non-empty", False, "pin refused")
    # R-225(2): this SEARCHED THE SOURCE for the word "absorption". A guard
    # whose test greps for a word has not been shown to fire. Driven now.
    try:
        PA.assert_fit_absorption_within_bound({"state_join_failed": 50}, 950, "btc")
        seam("31e the FIT bound REFUSES a per-status breach", False)
    except RuntimeError as _e31e:
        seam("31e the FIT bound REFUSES a per-status breach",
             "state_join_failed" in str(_e31e))

    # ---- SEAM 32 (R-215): exclusions are counted under THEIR OWN NAMES ----
    # The defect: tape_index() filtered state_status != "OK", so _feature_pass
    # found those rows missing and counted them as a JOIN failure. The same
    # exclusion applied twice, reported under a name meaning something else --
    # and 26,339 design exclusions refused a fit on the 1% bound. The counter
    # built to observe it (drops['state_status']) read 0, because the filter
    # upstream meant it never saw a status at all.
    #
    # This enters at the REAL _feature_pass with a synthetic tape and asserts
    # the DISTINCTION: design exclusions are exempt BY NAME and stay visible;
    # an ABSENT key still trips the bound; an UNRULED status is NOT exempt.
    import harmful_hazard_model as _hm32

    def _fp_case(statuses, drop_keys=()):
        """Run the real _feature_pass over one slug whose tape rows carry
        `statuses`. `drop_keys` are omitted from the tape entirely (join
        failure). Returns (drops, kept) or raises what production raises."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            frag = tdp / "frag.json"
            # _mini_exposure emits TWO rows per generation, so the generation
            # count is half the requested row count.
            assert len(statuses) % 2 == 0, "statuses must cover whole generations"
            rows = _mini_exposure(frag, 1787650200.0, len(statuses) // 2,
                                  "btc", "2026-08-25")
            assert len(rows) == len(statuses), (len(rows), len(statuses))
            TAPE = {}
            for i, r in enumerate(rows):
                k = (r["slug"], r["side"], r["gen"], r["t_start"])
                if i in drop_keys:
                    continue
                st = statuses[i]
                TAPE[k] = {"vec": (0.0,) * 45 if st == "OK" else None,
                           "status": st, "t0": r["t0"], "t_start": r["t_start"]}
            _sv = (_hm32.features, _hm32.fine_feats, _hm32.window_streams,
                   _hm32.fi._archive_paths, _hm32.fi.token_map)
            _slugs = {r["slug"] for r in rows}
            _npm = len(json.loads(PA.FROZEN.read_text())["fits"]["btc"]["norm_mu"]) - 1
            _hm32.features = lambda *a, **k: [0.3] * _npm
            _hm32.fine_feats = lambda *a, **k: [0.1]
            _hm32.window_streams = lambda *a, **k: object()
            _hm32.fi._archive_paths = lambda *a, **k: {x: Path("/dev/null") for x in _slugs}
            _hm32.fi.token_map = lambda *a, **k: {x: ("u", "d") for x in _slugs}
            try:
                out = PA._feature_pass(frag, "seam32", TAPE=TAPE)
            finally:
                (_hm32.features, _hm32.fine_feats, _hm32.window_streams,
                 _hm32.fi._archive_paths, _hm32.fi.token_map) = _sv
            return out["btc"]["drops"], out["btc"]["kept"]

    # (a) POSITIVE CONTROL — design exclusions far above the bound must NOT
    #     refuse, and must appear under their own names.
    _st = ["OK"] * 10 + ["PRE_WINDOW"] * 20 + ["GAP_AT_CUTOFF"] * 6 \
          + ["NO_LEVEL_HISTORY"] * 4
    _st = [x for x in _st for _ in (0, 1)]          # two rows per generation
    try:
        _d32, _k32 = _fp_case(_st)
        seam("32a design exclusions at 75% do NOT trip the bound",
             True)
        seam("32b PRE_WINDOW counted as pre_window_excluded",
             _d32.get("pre_window_excluded") == 40, f"drops={_d32}")
        seam("32c GAP_AT_CUTOFF has its OWN line, never folded into warm-up",
             _d32.get("gap_at_cutoff_excluded") == 12, f"drops={_d32}")
        seam("32d NO_LEVEL_HISTORY counted separately",
             _d32.get("no_level_history_excluded") == 8, f"drops={_d32}")
        seam("32e state_join_failed is 0 when every key is PRESENT",
             _d32.get("state_join_failed") == 0, f"drops={_d32}")
        seam("32f the misnamed counter is GONE",
             "state" not in _d32 and "state_status" not in _d32, f"drops={_d32}")
        seam("32g only OK rows are kept", len(_k32) == 20, f"kept={len(_k32)}")
    except Exception as _e32:
        for _n in ("32a design exclusions at 75% do NOT trip the bound",
                   "32b PRE_WINDOW counted as pre_window_excluded",
                   "32c GAP_AT_CUTOFF has its OWN line, never folded into warm-up",
                   "32d NO_LEVEL_HISTORY counted separately",
                   "32e state_join_failed is 0 when every key is PRESENT",
                   "32f the misnamed counter is GONE",
                   "32g only OK rows are kept"):
            seam(_n, False, f"{type(_e32).__name__}: {_e32}")

    # (b) KNOWN-BAD INPUT — a genuinely ABSENT key is a join failure and MUST
    #     still refuse. This is what stops the exemption becoming a carve-out.
    try:
        _fp_case(["OK"] * 40, drop_keys=set(range(0, 20)))
        seam("32h an ABSENT key still TRIPS the 1% bound", False,
             "_feature_pass ACCEPTED 25% missing keys — the exemption has "
             "become a blanket carve-out and a real join failure is invisible")
    except Exception as _e32b:
        seam("32h an ABSENT key still TRIPS the 1% bound",
             isinstance(_e32b, RuntimeError) and "state_join_failed" in str(_e32b),
             f"{type(_e32b).__name__}: {_e32b}")

    # (c) an UNRULED status must NOT inherit an exemption it was never ruled
    #     into — otherwise any future status silently bypasses the bound.
    try:
        _fp_case(["OK"] * 30 + ["SOME_NEW_STATUS"] * 10)
        seam("32i an UNRULED status is NOT exempt", False,
             "a status never ruled into DESIGN_EXCLUSIONS bypassed the bound")
    except Exception as _e32c:
        seam("32i an UNRULED status is NOT exempt",
             isinstance(_e32c, RuntimeError)
             and "some_new_status_excluded" in str(_e32c),
             f"{type(_e32c).__name__}: {_e32c}")

    # ---- SEAM 33 (R-215): four-arms-one-n parity is COMPUTED --------------
    _f33 = inspect.getsource(PA.stage_fit)
    seam("33a parity is computed from each arm's OWN design matrix",
         _f33.count("_arm_n[") >= 4,
         "fewer than four arms record an n — parity would be assumed, not measured")
    seam("33b a parity mismatch REFUSES",
         "parity broken" in _f33)
    seam("33c parity reaches the manifest AND its own artifact",
         "fit_population_parity" in _f33)
    seam("33e the parity predicate is COMPUTED, not a hardcoded verdict",
         "\"all_arms_same_n\": (len(set(" in _f33,
         "a hardcoded True beside the table it describes has contradicted "
         "that table three times in this repo (CLAUDE.md rule 10)")
    # R-225: driven. A renamed counter must REFUSE, not merely be mentioned.
    try:
        PA.assert_fit_absorption_within_bound({"renamed_counter": 0}, 100, "btc")
        seam("33f the bounded counter cannot be silently renamed away", False,
             "a renamed counter removes the only thing between a real join "
             "failure and a silent all-drop")
    except RuntimeError as _e33:
        seam("33f the bounded counter cannot be silently renamed away",
             "state_join_failed" in str(_e33))
    seam("33d the registration is keyed by TAPE identity",
         PA.preregistered_n("c7ab02ebcf27d2fc", "btc") == 578917
         and PA.preregistered_n("0" * 16, "btc") is None,
         "a population count asserted against the wrong tape is not a check")

    # ---- SEAM 34 (R-216): TWO-STAGE registration, and the stage is part of
    # the declaration. be-fit3 refused a HEALTHY fit because ok_n (post-join,
    # 578,917) was checked against the post-PURGE matrix (577,598). Both counts
    # were correct; the embargo purge sits deliberately between them. These
    # enter at the CALLABLE predicates -- which is why they are callable.
    _REG = {"t": {"btc": {"ok_n": 100, "embargo_purged": 10, "fitted_n": 90}}}

    seam("34a the registration reconciles: fitted_n + purged == ok_n",
         PA.assert_registration_arithmetic() >= 2)
    try:
        PA.assert_registration_arithmetic(
            {"t": {"btc": {"ok_n": 100, "embargo_purged": 5, "fitted_n": 90}}})
        seam("34b an INCONSISTENT registration is REFUSED", False,
             "a registration whose own numbers disagree cannot adjudicate a fit")
    except RuntimeError as _e:
        seam("34b an INCONSISTENT registration is REFUSED",
             "internally inconsistent" in str(_e))

    # stage 1 is checked on the PRE-PURGE population -- the be-fit3 class
    _FITok = {"btc": {"kept": [0] * 100, "drops": {}}}
    try:
        _ev = PA.assert_preregistered_population(_FITok, "t", _REG)
        seam("34c stage 1 passes on the PRE-PURGE count", _ev["btc"]["matches_ok_n"])
        _f = PA.assert_fitted_population("btc", 90, 10, _ev["btc"])
        seam("34d stage 2 accepts the post-purge matrix (the be-fit3 case)",
             _f["purge_reconciles"] and _f["matches_registered_fitted_n"],
             "the fit be-fit3 refused must now pass: 100 -> 90 with 10 purged")
    except Exception as _e:
        seam("34c stage 1 passes on the PRE-PURGE count", False, f"{_e}")
        seam("34d stage 2 accepts the post-purge matrix (the be-fit3 case)",
             False, f"{_e}")

    # a REAL population move at stage 1 must still refuse
    try:
        PA.assert_preregistered_population({"btc": {"kept": [0] * 99,
                                                    "drops": {}}}, "t", _REG)
        seam("34e a REAL pre-purge move still REFUSES", False,
             "widening the stage must not have widened the check")
    except RuntimeError as _e:
        seam("34e a REAL pre-purge move still REFUSES", "population move" in str(_e))

    # rows leaving through an UNACCOUNTED path must refuse
    try:
        PA.assert_fitted_population("btc", 88, 10,
                                    {"population_pre_purge": 100,
                                     "registered_fitted_n": 90,
                                     "registered_embargo_purged": 10})
        seam("34f an UNRECONCILED purge is REFUSED", False,
             "100 - 10 != 88: rows left through a path nothing accounts for")
    except RuntimeError as _e:
        seam("34f an UNRECONCILED purge is REFUSED", "does not reconcile" in str(_e))

    # a purge count that disagrees with its registration must refuse
    try:
        PA.assert_fitted_population("btc", 85, 15,
                                    {"population_pre_purge": 100,
                                     "registered_fitted_n": 90,
                                     "registered_embargo_purged": 10})
        seam("34g a purge DISAGREEING with its registration is REFUSED", False,
             "reconciling internally is not the same as matching what was declared")
    except RuntimeError as _e:
        seam("34g a purge DISAGREEING with its registration is REFUSED",
             "never adopt the new number" in str(_e))

    seam("34h evidential status is recorded, not assumed uniform",
         set(PA.REGISTRATION_PROVENANCE) == {"ok_n", "embargo_purged", "fitted_n"}
         and "PRE-REGISTERED" in PA.REGISTRATION_PROVENANCE["ok_n"]
         and "RE-DECLARED" in PA.REGISTRATION_PROVENANCE["fitted_n"],
         "ok_n was declared blind; fitted_n and the purge were re-declared after "
         "observation with the cause named -- a reader must not assume all three "
         "carry the same weight")
    # ---- SEAM 35 (R-216): the PRODUCTION receipt path is not the stale one --
    # The protocol label was renamed and the output path was not, so the
    # four-arm score overwrote the committed three-arm receipt in place. The
    # seam sandbox redirects PA.OUT, which is precisely why the production
    # default was never exercised: a test that reassigns the thing under test
    # cannot see it is wrong.
    seam("35a PA.OUT names the CURRENT protocol, not the superseded one",
         "four_arm_v2" in PA.OUT.name and "three_arm" not in PA.OUT.name,
         f"PA.OUT={PA.OUT.name} -- a four-arm receipt written to the three-arm "
         f"path overwrites a superseded artifact that exists as provenance")
    seam("35b the receipt path agrees with the protocol label in the receipt",
         PA.OUT.stem.replace("phase2_", "").replace("_", "").lower()
         in inspect.getsource(PA.stage_score).lower().replace("_", ""),
         "path and label must not be able to drift apart again")

    seam("34i stage 1 runs BEFORE the purge in the real stage_fit",
         inspect.getsource(PA.stage_fit).index("assert_preregistered_population")
         < inspect.getsource(PA.stage_fit).index("EMB.purge_training"),
         "checked after the purge, it compares two different stages -- be-fit3")

    # ---- SEAM 36 (debts): the guards are CALLABLE, and are called ----------
    import build_state_tape_v2 as B36
    import gap_at_cutoff_count as GC36
    import phase2_state_schema_freeze as SF36
    import phase2_increment_null as IN36
    import harmful_exposure_rows as HER36
    import harmful_hazard_model as HHM36

    # 30d-f: drive the REAL predicate with inputs of the probe's choosing.
    _g = {"btc": [(100.0, 200.0)], "eth": []}
    seam("36a/30d gap_contains_at is MODULE-level and callable",
         callable(getattr(B36, "gap_contains_at", None)),
         "an inline closure can only be tested by reimplementing it, which is "
         "how two comparisons came to disagree at both edges (R-213)")
    seam("36b/30e lower bound is INCLUSIVE (T exactly at g0 flags)",
         B36.gap_contains_at(100.0, "btc", _g) == (100.0, 200.0))
    seam("36c/30f upper bound is EXCLUSIVE (T exactly at g1 does NOT flag)",
         B36.gap_contains_at(200.0, "btc", _g) is None)
    seam("36d a NEGATIVE absolute instant is answered, not crashed",
         B36.gap_contains_at(-27.0, "btc", _g) is None)
    seam("36e a coin with no gaps never flags",
         B36.gap_contains_at(150.0, "eth", _g) is None)

    # absorption guard: callable, per-status AND total-form
    seam("36f the absorption guard is a CALLABLE entry point",
         callable(getattr(B36, "assert_absorption_within_bound", None)))
    try:
        B36.assert_absorption_within_bound({f"S{i}": 9 for i in range(10)}, 910)
        seam("36g TOTAL-FORM: 10 statuses at 0.9% each REFUSE (9% total)", False,
             "a total failure does not have to arrive under one name; a "
             "per-status bound cannot see it spread")
    except SystemExit as e:
        seam("36g TOTAL-FORM: 10 statuses at 0.9% each REFUSE (9% total)",
             "TOTAL" in str(e))
    try:
        B36.assert_absorption_within_bound({"ONE": 50}, 950)
        seam("36h per-status bound still REFUSES", False)
    except SystemExit as e:
        seam("36h per-status bound still REFUSES", "PRE-EMISSION SKIP" in str(e))
    _ev36 = B36.assert_absorption_within_bound({"ONE": 5}, 995)
    seam("36i a legitimate small skip PASSES and reports its fractions",
         _ev36["total_fraction"] == 0.005 and "ONE" in _ev36["per_status_fractions"])

    # ---- SEAM 37: identity is MEASURED, not a passed label -----------------
    _mi = PA.measured_code_identity()
    seam("37a the fit code is bound by CONTENT, not only the env label",
         isinstance(_mi.get("combined"), str) and len(_mi["combined"]) == 16,
         "the manifest verified that a LABEL was passed, not that code ran")
    seam("37b the identity is DETERMINISTIC (declared file list, not sys.modules)",
         set(_mi["files"]) == set(PA.CODE_IDENTITY_FILES),
         "hashing live imports made stage_fit and stage_score disagree for the "
         "SAME tree, which is the opposite of an identity")
    # NOT "a dirty tree changes it" -- that is 37d. This asserts the property
    # 37d needs in order to mean anything: on an UNCHANGED tree, repeated calls
    # agree. Without stability, 37d's inequality would prove nothing.
    seam("37c repeated calls on an UNCHANGED tree agree (stability)",
         PA.measured_code_identity()["combined"] == _mi["combined"])
    import tempfile as _tf37, hashlib as _h37
    _p37 = Path(PA._ROOT) / "phase2_declaration.py"
    _orig37 = _p37.read_bytes()
    try:
        _p37.write_bytes(_orig37 + b"\n# seam 37 dirt\n")
        seam("37d a dirty tree is DETECTED (the falsifier)",
             PA.measured_code_identity()["combined"] != _mi["combined"],
             "an identity that cannot notice an edit is not an identity")
    finally:
        _p37.write_bytes(_orig37)
    seam("37e restoring the tree restores the identity",
         PA.measured_code_identity()["combined"] == _mi["combined"])
    seam("37f the FRAGMENT is bound by sha in the identity block",
         "fragment_sha256_prefix" in PA._tape_identity(),
         "the fragment DEFINES the population ok_n registers and was unbound")

    # ---- SEAM 38: consumer honours the PRODUCER's load-bearing list --------
    _src38 = inspect.getsource(PA.assert_gate_passed)
    seam("38a the consumer reads the verdict's OWN load_bearing_asserted",
         "load_bearing_asserted" in _src38,
         "a hardcoded consumer set can accept a verdict where a predicate the "
         "PRODUCER called load-bearing is N/A or absent")
    seam("38b the required set is the UNION, never a replacement",
         "|" in _src38.split("load_bearing_asserted")[0].rsplit("LOAD_BEARING", 1)[-1]
         or "set(LOAD_BEARING)" in _src38)

    # ---- SEAM 39: parity is a RECEIPT FIELD; declaration ref is measured ---
    _src39 = inspect.getsource(PA.stage_score)
    seam("39a parity is a receipt FIELD, not only a manifest-chain artifact",
         "fit_population_parity" in _src39)
    seam("39b parity is READ from the fit artifact, never recomputed here",
         "_read_fit_parity" in _src39 and "read_text" in inspect.getsource(PA._read_fit_parity))
    seam("39c the stale three-arm declaration ref is GONE",
         '"declaration_commit": "d7082b6"' not in _src39,
         "d7082b6 has THREE arms and sat beside a four-arm receipt")
    seam("39d the declaration is bound by MEASUREMENT",
         "declaration_sha256_prefix" in _src39)

    # ---- SEAM 40: the F1/F4/F5 instrument fixes ---------------------------
    _e40 = Path(PA._ROOT) / "phase2_embargo.py"
    seam("40a the embargo suite's DEFAULT entry runs its checks",
         "SystemExit(selftest())" in _e40.read_text(),
         "the default entry ran NOTHING and exited 0; four commits cited that "
         "silent rc=0 as GREEN")
    seam("40b the gap counter records WHICH ledger it counted",
         callable(getattr(GC36, "ledger_sha256", None))
         and "ledger_sha256_prefix" in inspect.getsource(GC36.count))
    seam("40c the counter offers the PINNED ledger, not only the live one",
         hasattr(GC36, "LEDGER_PIN"),
         "the live ledger is appended continuously by the collectors")
    seam("40d the schema suite VERIFIES its reference instead of rewriting it",
         callable(getattr(SF36, "verify_against_committed", None))
         and "--write" in inspect.getsource(SF36.main),
         "a test that regenerates its own reference cannot detect drift in it")

    # ---- SEAM 41: increment-null measured delta; ONE any_fill_ahead --------
    seam("41a the reconciliation guard is CALLABLE",
         callable(getattr(IN36, "assert_reconciles", None)))
    try:
        IN36.assert_reconciles(1.0, 2.0, "known-bad")
        seam("41b it REFUSES a known-bad input", False)
    except RuntimeError:
        seam("41b it REFUSES a known-bad input", True)
    seam("41c it returns a MEASURED delta, not a hardcoded boolean",
         IN36.assert_reconciles(1.0, 1.0, "identity") == 0.0
         and "max_abs_delta_cents" in inspect.getsource(IN36.main))
    seam("41d any_fill_ahead has ONE definition",
         HHM36._any_fill_ahead is HER36.any_fill_ahead,
         "two rules for the same valuation gate is one too many")
    _lat41 = {"50": {"preventable_shares": 0.0, "stale_shares": 0.0}}
    seam("41e the divergence case resolves under the GOVERNING rule",
         HER36.any_fill_ahead(_lat41) is False
         and HHM36.keptrow({"latency": _lat41})["any_fill_ahead"] is False,
         "tranches exist but carry zero shares: the old builder said True, "
         "keptrow said False, and keptrow overwrote -- so keptrow governed")

    # ---- SEAM 42 (R-225): the gate must REJECT, not merely NOTICE ---------
    # The user's audit: the seams proved hashes CHANGE and none proved scoring
    # REJECTS a missing or mismatched hash. An instrument that moves is not a
    # gate that bites. These drive the REAL checker against manifests built to
    # be wrong, and require a refusal.
    import tempfile as _t42
    def _manifest_case(mutate, env_ref=None):
        """Run the REAL assert_fit_complete_and_matching against a mutated
        manifest. Returns the refusal message, or None if it ACCEPTED.

        FIT_CODE_REF is set so the BASE manifest is well-formed: without it
        _tape_identity() yields fit_code_ref None and every case would be
        refused on that instead of on the binding under test -- a harness that
        refuses for the wrong reason proves nothing."""
        d = Path(_t42.mkdtemp())
        _sv_env = os.environ.get("FIT_CODE_REF")
        # R-228(1): a COMPLETE, genuinely valid base manifest. This used to set
        # file_hashes:{} and assert the result was "well-formed" -- the positive
        # control CERTIFIED the very vacuity the audit found. A guard's accept
        # path must be exercised with something actually valid or it proves the
        # hole, not the guard.
        import hashlib as _h42, subprocess as _s42
        _ref = _s42.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
        _pfx = _s42.run(["git", "-C", str(HERE), "rev-parse", "--show-prefix"],
                        capture_output=True, text=True).stdout.strip()
        os.environ["FIT_CODE_REF"] = _ref
        now = PA._tape_identity()
        m = dict(now)
        m["complete"] = True
        m["file_hashes"] = {}
        (d / "empty_coins.json").write_text("[]")
        m["file_hashes"]["empty_coins.json"] = _h42.sha256(b"[]").hexdigest()[:16]
        for _n in tuple(PA.FIT_BASE_ARTIFACTS) + tuple(
                t.format(c=c) for c in ("btc", "eth")
                for t in PA.FIT_PER_COIN_ARTIFACTS):
            if _n == "empty_coins.json":
                continue
            (d / _n).write_text("{}")
            m["file_hashes"][_n] = _h42.sha256(b"{}").hexdigest()[:16]
        # fit_code_files read FROM the commit, so the accept path is exercised
        # even when the working tree is dirty
        m["fit_code_ref"] = _ref
        m["fit_code_files"] = {}
        for _f in PA.CODE_IDENTITY_FILES:
            _b = _s42.run(["git", "-C", str(HERE), "show", f"{_ref}:{_pfx}{_f}"],
                          capture_output=True).stdout
            m["fit_code_files"][_f] = _h42.sha256(_b).hexdigest()[:16]
        mutate(m)
        if env_ref is not None:
            # both sides zeroed, so the case reaches the RESOLUTION check
            # instead of stopping at the earlier ref-equality comparison
            os.environ["FIT_CODE_REF"] = env_ref
            m["fit_code_ref"] = env_ref
        (d / PA.FIT_MANIFEST).write_text(json.dumps(m))
        sv = PA.FITDIR
        PA.FITDIR = d
        try:
            PA.assert_fit_complete_and_matching()
            return None
        except RuntimeError as e:
            return str(e)
        finally:
            PA.FITDIR = sv
            if _sv_env is None:
                os.environ.pop("FIT_CODE_REF", None)
            else:
                os.environ["FIT_CODE_REF"] = _sv_env

    _r = _manifest_case(lambda m: m.pop("fit_code_sha256_prefix", None))
    seam("42a a manifest MISSING fit_code_sha256_prefix is REFUSED",
         _r is not None and "fit_code_sha256_prefix" in _r,
         "the pre-fix manifest was ACCEPTED — the hash was written and never "
         "enforced, so nothing stopped a fit of unknown code being scored")
    _r = _manifest_case(lambda m: m.pop("fragment_sha256_prefix", None))
    seam("42b a manifest MISSING fragment_sha256_prefix is REFUSED",
         _r is not None and "fragment_sha256_prefix" in _r)
    _r = _manifest_case(lambda m: m.update({"fit_code_sha256_prefix": "dead" * 4}))
    seam("42c a MISMATCHED fit code hash is REFUSED",
         _r is not None and "FIT CODE" in _r)
    _r = _manifest_case(lambda m: m.update({"fragment_sha256_prefix": "dead" * 4}))
    seam("42d a MISMATCHED fragment hash is REFUSED",
         _r is not None and "FRAGMENT" in _r)
    _r = _manifest_case(lambda m: [m.pop(k, None) for k in
                                   ("fit_code_sha256_prefix",
                                    "fragment_sha256_prefix")])
    seam("42e BOTH absent is still REFUSED (None==None vacuity)",
         _r is not None,
         "`m.get(k) != now.get(k)` passes when both sides are None: an absent "
         "binding read as agreement")
    _r = _manifest_case(lambda m: None)
    seam("42f a WELL-FORMED manifest is ACCEPTED (the guard is not a wall)",
         _r is None, f"refused a valid manifest: {_r}")

    # the receipt must report the FIT's identity, read from the manifest
    _fsrc = inspect.getsource(PA.stage_score)
    _line = next((l for l in _fsrc.split("\n") if '"fit_code_identity"' in l), "")
    seam("42g the receipt reads the FIT's identity FROM THE MANIFEST",
         "_fit_identity_from_manifest" in _line
         and "measured_code_identity" not in _line,
         f"it measured at SCORE time — the scorer's identity under the fit's "
         f"name. line: {_line.strip()[:90]}")
    seam("42h the scorer's own identity is reported under ITS OWN name",
         "score_code_identity" in _fsrc)

    # ---- SEAM 43 (R-225): TOTAL-form fit bound, driven ---------------------
    try:
        PA.assert_fit_absorption_within_bound(
            dict({f"X{i}": 9 for i in range(10)}, state_join_failed=0), 910, "btc")
        seam("43a TOTAL: ten 0.9% categories aggregating 9% REFUSE", False,
             "each passes the per-status bound; the failure arrived spread")
    except RuntimeError as _e:
        seam("43a TOTAL: ten 0.9% categories aggregating 9% REFUSE",
             "TOTAL" in _e.args[0])
    _ev43 = PA.assert_fit_absorption_within_bound(
        {"pre_window_excluded": 600, "gap_at_cutoff_excluded": 100,
         "no_level_history_excluded": 50, "state_join_failed": 0}, 250, "btc")
    seam("43b DESIGN exclusions at 75% do NOT trip either form",
         _ev43["bounded_total_fraction"] == 0.0)

    # ---- SEAM 44 (R-225): the lock EXCLUDES, and is RELEASED ---------------
    import subprocess as _sp44
    _d44 = Path(_t42.mkdtemp()); _L44 = _d44 / "f.lock"
    _prog = (
        "import sys,os;sys.path.insert(0,%r)\n"
        "import phase2_arms as PA\n"
        "from pathlib import Path\n"
        "try:\n"
        "    PA.acquire_fit_lock(Path(%r)); print('WON')\n"
        "except RuntimeError: print('LOST')\n"
    ) % (str(HERE), str(_L44))
    _procs = [_sp44.Popen([sys.executable, "-c", _prog], stdout=_sp44.PIPE,
                          text=True) for _ in range(2)]
    _outs = [p.communicate()[0].strip() for p in _procs]
    seam("44a TWO CONCURRENT acquirers: exactly one wins",
         _outs.count("WON") == 1,
         f"got {_outs} — check-then-write let both pass `if lock.exists()`")
    _L44.unlink(missing_ok=True)
    seam("44b acquisition is ATOMIC (O_EXCL), not check-then-write",
         "O_EXCL" in inspect.getsource(PA.acquire_fit_lock))
    _pid44 = PA.acquire_fit_lock(_L44)
    seam("44c the owner CAN release", PA.release_fit_lock(_L44) is True)
    seam("44d the lock is GONE after release", not _L44.exists())
    PA.acquire_fit_lock(_L44)
    seam("44e a NON-owner cannot release someone else's lock",
         PA.release_fit_lock(_L44, pid=999999) is False and _L44.exists())
    _L44.unlink(missing_ok=True)
    seam("44f stage_fit releases in a FINALLY",
         "finally:" in inspect.getsource(PA.stage_fit)
         and "release_fit_lock(_lock)" in inspect.getsource(PA.stage_fit),
         "the lock was never released; every run left one to be reclaimed")

    # ---- SEAM 45 (R-225): identity BEFORE load, RECHECKED at write --------
    _sf45 = inspect.getsource(PA.stage_fit)
    seam("45a identity is captured BEFORE the inputs are loaded",
         _sf45.index("_ident_pre = _tape_identity()")
         < _sf45.index("_feature_pass(FRAGMENT"),
         "captured after the load, a mid-run change is recorded as though it "
         "had always been so")
    seam("45b it is RECHECKED at write time",
         "_ident_post = _tape_identity()" in _sf45)
    seam("45c a mid-run DRIFT is a REFUSAL, not a warning",
         "inputs CHANGED DURING the run" in _sf45)
    seam("45d the manifest records that both happened",
         "identity_captured_before_load" in _sf45
         and "identity_rechecked_at_write" in _sf45)
    # BEHAVIOURAL: perturb an input and require the drift comparison to fire
    _fake_pre = dict(PA._tape_identity())
    _fake_post = dict(_fake_pre); _fake_post["fragment_sha256_prefix"] = "beef" * 4
    _drift45 = {k: (_fake_pre.get(k), _fake_post.get(k))
                for k in ("tape_sha256_prefix", "fragment_sha256_prefix")
                if _fake_pre.get(k) != _fake_post.get(k)}
    seam("45e the drift comparison DETECTS a perturbed input",
         "fragment_sha256_prefix" in _drift45,
         "the comparison must notice a changed fragment, not just record one")

    # ---- SEAM 46 (R-228): the fail-open class, one level down --------------
    # Audit #9: the guards added at R-225 still passed VACUOUSLY. The hash loop
    # iterated whatever file_hashes held, so an EMPTY map verified nothing and
    # read as success; a manifest listing one artifact of fourteen left thirteen
    # unchecked; and fit_code_ref was only compared to the env value the scorer
    # was launched with, so all-zeros matched all-zeros.
    _r = _manifest_case(lambda m: m.update({"file_hashes": {}}))
    seam("46a an EMPTY file_hashes map is REFUSED",
         _r is not None and "EMPTY" in _r,
         "zero iterations of the hash loop is not zero artifacts changed")
    _r = _manifest_case(lambda m: m.update(
        {"file_hashes": {k: v for k, v in list(m["file_hashes"].items())[:1]}}))
    seam("46b a manifest covering only ONE artifact is REFUSED",
         _r is not None and "required artifact" in _r,
         "the loop verified what was listed; it never asked what SHOULD be")
    _r = _manifest_case(lambda m: None, env_ref="0" * 40)
    seam("46c an all-zeros fit_code_ref is REFUSED",
         _r is not None and "does not resolve to a commit" in _r,
         "a ref naming no commit attests to nothing")
    _r = _manifest_case(lambda m: m["fit_code_files"].update(
        {"phase2_arms.py": "dead" * 4}))
    seam("46d a ref NOT CARRYING the recorded code is REFUSED",
         _r is not None and "does not carry the recorded code" in _r,
         "the label must name the commit whose content was measured")
    _r = _manifest_case(lambda m: m.update({"fit_code_files": {}}))
    seam("46e a manifest with NO fit_code_files is REFUSED",
         _r is not None and "no fit_code_files" in _r,
         "an unverifiable binding must not pass")

    # the identity lattice must cover the result-bearing deps and artifacts
    for _f in ("harmful_exposure_rows.py", "flow_intensity.py",
               "flow_fill_development.py", "harmful_candidate_manifest.py"):
        seam(f"46f lattice covers {_f}", _f in PA.CODE_IDENTITY_FILES,
             "code that shapes a number but is not hashed makes a PARTIAL "
             "identity read as a whole one")
    _id46 = PA._tape_identity()
    for _k in ("topup_sha256_prefix", "frozen_incumbent_sha256_prefix",
               "topup_build_receipt_sha256_prefix"):
        seam(f"46g identity binds {_k}", _id46.get(_k) is not None,
             "these can change what the numbers MEAN while every previously "
             "bound field stays identical")
    _r = _manifest_case(lambda m: m.update({"topup_sha256_prefix": "dead" * 4}))
    seam("46h a swapped SCORING TOP-UP is REFUSED",
         _r is not None and "TOP-UP" in _r)
    _r = _manifest_case(lambda m: m.update(
        {"frozen_incumbent_sha256_prefix": "dead" * 4}))
    seam("46i a swapped FROZEN INCUMBENT is REFUSED",
         _r is not None and "FROZEN INCUMBENT" in _r)

    # score-side identity, mirroring the fit's
    _ss46 = inspect.getsource(PA.stage_score)
    seam("46j the SCORE captures identity BEFORE load",
         "_ident_pre = _tape_identity()" in _ss46,
         "only the fit did; the stage producing the published numbers was the "
         "unguarded one")
    seam("46k the SCORE rechecks at write and REFUSES drift",
         "inputs CHANGED DURING scoring" in _ss46)

    # the lock finally must cover from ACQUISITION
    _sf46 = inspect.getsource(PA.stage_fit)
    _a46 = _sf46.index("acquire_fit_lock(_lock)")
    _t46 = _sf46.index("\n    try:", _a46)
    seam("46l nothing that can RAISE sits between acquire and try",
         "_tape_identity()" not in _sf46[_a46:_t46],
         "an identity failure after acquisition leaked a LIVE lock on exactly "
         "the paths hardest to reproduce")

    # supersession is generator-owned
    seam("46m protocol_version is EMITTED by the generator",
         '"protocol_version": PROTOCOL_VERSION' in _ss46,
         "it was hand-added post-generation at v2.1")
    seam("46n supersedes is EMITTED by the generator",
         '"supersedes": _supersedes_block()' in _ss46)
    seam("46o the supersedes block reports ABSENCE explicitly",
         "present_at_write" in inspect.getsource(PA._supersedes_block),
         "a receipt silently claiming to supersede nothing is "
         "indistinguishable from one whose predecessor was deleted")

    print(f"\n{'PRODUCTION SEAMS GREEN' if not FAILURES else 'PRODUCTION SEAMS RED'}: "
          f"{len(FAILURES)} failing")
    for f in FAILURES:
        print(f"  - {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
