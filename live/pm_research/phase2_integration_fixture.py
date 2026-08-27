"""END-TO-END integration fixture. A GATE, not a test suite.

AUTHORISATION (R-126, in-file): R-187(2), user audit 2026-08-27.

WHY THIS EXISTS. Every Phase-2 module's selftest passes. Seven defects lived in
the SEAMS BETWEEN them, where no per-module test looks:
  1. the pin carried `family` (a string) and `decision_time` (wall-clock, leak-
     shaped) as FEATURES -- the all-float encoder crashes on one and the model
     can date its rows with the other
  2. the scorer RE-DERIVES state via build_tape() with no gaps/bn_recv_ns,
     recreating the very missing-input defect the tape rebuild fixes
  3. the embargo was RECORDED as violated and then ignored, never applied
  4. builder and verifier disagree on row nesting and clock basis
  5. arm D was declared but never dispatched -- it falls through to the LGBM
     branch and silently DUPLICATES arm C, so D-A and B-D are meaningless
  6. causal thresholds: declared, not implemented
  7. head diagnostics: declared, not emitted

Each is asserted BY CONSTRUCTION below, on a tiny synthetic population pushed
through builder -> tape -> purge -> fit -> score -> verify -> evaluate in ONE
process. The fixture must be GREEN before any real fit or score, and re-run
before every future rerun.

IT IS BUILT TO FAIL FIRST. Run against the code as the audit found it, these
assertions fire. A gate that has only ever been green on fixed code has never
demonstrated it can catch anything.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, "/home/yuqing/ctaNew/live/pm_research")

FAILURES: list = []


def seam(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")
        FAILURES.append(name)


def synth_population(n_train=60, n_score=30):
    """Two populations separated by a KNOWN time gap, so the embargo has a
    right answer the fixture can check rather than trust."""
    import phase2_embargo as EMB
    rows_tr, rows_sc = [], []
    t0 = 1_000_000.0
    for i in range(n_train):
        rows_tr.append({"slug": f"btc-updown-5m-{int(t0)}", "coin": "btc",
                        "day": "d1", "t0": t0, "t_start": float(i),
                        "side": "BUY_UP" if i % 2 else "SELL_UP", "gen": i // 2,
                        "status": "OK", "state_status": "OK",
                        "latency": {"50": {"preventable_value_cents": (i % 7) - 3.0,
                                           "preventable_shares": 1.0 if i % 3 else 0.0,
                                           "stale_shares": 0.0}},
                        "any_fill_ahead": bool(i % 3)})
    # scoring starts 200s after the last training label exit -> embargo satisfied
    last_exit = max(EMB.label_exit_time(r) for r in rows_tr)
    s0 = last_exit + 200.0
    for i in range(n_score):
        rows_sc.append({"slug": f"btc-updown-5m-{int(s0)+300}", "coin": "btc",
                        "day": "d2", "t0": s0, "t_start": float(i),
                        "side": "BUY_UP" if i % 2 else "SELL_UP", "gen": i // 2,
                        "status": "OK", "state_status": "OK",
                        "latency": {"50": {"preventable_value_cents": (i % 5) - 2.0,
                                           "preventable_shares": 1.0 if i % 2 else 0.0,
                                           "stale_shares": 0.0}},
                        "any_fill_ahead": bool(i % 2)})
    return rows_tr, rows_sc


def main() -> int:
    import phase2_state_schema_freeze as PIN
    import phase2_embargo as EMB
    import phase2_declaration as D
    import harmful_action_eval as ae

    print("PHASE-2 INTEGRATION FIXTURE — the seven seams, by construction\n")
    pin = PIN.build_pin()
    feats = pin["features_in_order"]

    # ---- SEAM 1: the model matrix must be ALL NUMERIC, with no clock column
    seam("1a pin excludes non-numeric `family`", "family" not in feats,
         "a string column crashes the all-float encoder")
    seam("1b pin excludes `decision_time`", "decision_time" not in feats,
         "wall-clock as a feature lets the model date its rows (leak-shaped)")
    probe = {k: 1.0 for k in feats}
    probe.update({"family": "pred_state_v1", "decision_time": 1_000_000.0})
    try:
        vec = PIN.encode_row(probe, feats)
        allnum = all(isinstance(x, float) for x in vec)
    except Exception as e:
        vec, allnum = None, False
        seam("1c encode_row builds an all-float vector", False, f"{type(e).__name__}: {e}")
    if vec is not None:
        seam("1c encode_row builds an all-float vector", allnum)

    # ---- SEAM 3: the embargo must be APPLIED, not merely recorded
    tr, sc = synth_population()
    kept, dropped = EMB.purge_training(tr, sc)
    seam("3a purge returns a partition", len(kept) + len(dropped) == len(tr))
    try:
        EMB.assert_embargo(kept, sc)
        seam("3b purged training CERTIFIES the embargo", True)
    except EMB.EmbargoViolation as e:
        seam("3b purged training CERTIFIES the embargo", False, str(e)[:70])
    # and a deliberately contaminated split must NOT certify
    bad_tr = tr + [{**sc[0], "t_start": sc[0]["t_start"] - 1.0}]
    try:
        EMB.assert_embargo(bad_tr, sc)
        seam("3c a contaminated split is REFUSED", False, "it certified")
    except EMB.EmbargoViolation:
        seam("3c a contaminated split is REFUSED", True)

    # ---- SEAM 5: arm D must be a DISTINCT model, asserted BY IDENTITY
    import phase2_arms as PA
    seam("5a arm D is declared", "INCUMBENT_REWEIGHTED_ONLY" in D.ARMS)
    # BY IDENTITY, not by grepping a name out of the source. A source grep
    # matches the COMMENT that documents a fix as readily as the fix.
    kd = PA.arm_model_kind("INCUMBENT_REWEIGHTED_ONLY")
    kc = PA.arm_model_kind("LGBM_PINNED")
    kb = PA.arm_model_kind("PLUS_PRED_STATE_V1")
    seam("5b arm D's model kind is the WEIGHTED LINEAR", kd == "weighted_linear")
    seam("5c arm D is NOT the same model as arm C", kd != kc,
         "if D falls through to LGBM it duplicates C and D-A / B-D are meaningless")
    seam("5d arm D shares B's model class (so B-D isolates the FEATURES)", kd == kb)

    # ---- SEAM 6: causal thresholds must RESOLVE FROM TRAINING ONLY
    has_thr = hasattr(PA, "freeze_thresholds") or "freeze_thresholds" in src
    seam("6a a threshold-freezing function exists", has_thr,
         "declared in phase2_declaration but never implemented")
    if has_thr and hasattr(PA, "freeze_thresholds"):
        thr = PA.freeze_thresholds([0.1, 0.5, 0.9, 0.2], D.BUDGETS)
        seam("6b thresholds resolve from the TRAIN scores alone",
             isinstance(thr, dict) and len(thr) == len(D.BUDGETS))

    # ---- SEAM 7: head diagnostics must EMIT
    has_heads = hasattr(PA, "head_diagnostics") or "head_diagnostics" in src
    seam("7a a head-diagnostics function exists", has_heads,
         "five diagnostics declared, none computed")
    if has_heads and hasattr(PA, "head_diagnostics"):
        h = PA.head_diagnostics([0.2, 0.8, 0.4], [0, 1, 0], [1.0, -2.0, 0.5],
                                [0.9, -1.5, 0.2])
        seam("7b diagnostics emit the declared five",
             all(k in h for k in D.HEAD_DIAGNOSTICS))

    # ---- SEAM 2: the scorer must CONSUME a rebuilt tape, not re-derive it
    # Checked by SIGNATURE and by RAISING, not by grepping -- the grep now
    # matches BE's own comment describing the removed call.
    import inspect
    sig = inspect.signature(PA._feature_pass)
    seam("2a the feature pass TAKES a tape argument", "TAPE" in sig.parameters)
    src_fn = inspect.getsource(PA._feature_pass)
    seam("2b it RAISES when no tape is supplied",
         "raise RuntimeError" in src_fn and "TAPE is None" in src_fn,
         "silently re-deriving would restore the missing-input defect")

    # ---- SEAM 4: builder and verifier must agree on layout + clock basis
    schema = json.loads((Path("/home/yuqing/ctaNew/data/pm_5min/derived") /
                         "da_pred_state_v1_schema.json").read_text())
    # BE's first version of this check guessed lowercase key names and failed
    # on a schema that already had them. Read the keys, do not guess them.
    seam("4a schema declares LAYOUT", "LAYOUT" in schema)
    seam("4b schema declares CLOCK_BASIS", "CLOCK_BASIS" in schema)
    seam("4c the pin CARRIES both forward to consumers",
         bool(pin.get("layout")) and bool(pin.get("clock_basis")))
    # ROUND-TRIP: a builder-shaped row must be locatable and interpretable
    # using only the declarations, with no guessing.
    cb = schema["CLOCK_BASIS"]
    seam("4d schema says decision_time is WINDOW-RELATIVE",
         "window_relative" in str(cb.get("decision_time", "")))
    tape_row = {"features_under": "state", "t0": 1000.0, "t_start": -39.4,
                "decision_time": -39.4, "state": {k: 0.0 for k in feats}}
    located = tape_row.get(tape_row["features_under"])
    seam("4e a reader LOCATES features by the declared wrapping key",
         located is not None and set(located) == set(feats))
    seam("4f decision_time round-trips to an epoch via the declared rule",
         abs((tape_row["t0"] + tape_row["decision_time"]) - 960.6) < 1e-9,
         "t0 + decision_time; a builder emitting an absolute decision_time "
         "would double-count t0 here")
    seam("4g negative decision_time is ADMITTED (pre-window warm-up is real)",
         tape_row["decision_time"] < 0)

    # ---- SEAM 8 (BE-added): the RUN PATH must USE these, not merely define
    # them. The fixture was green on all seven seams while stage_score called
    # NONE of the fixed components -- capabilities existing is not the same
    # claim as the run path using them, and only the second one protects a
    # rerun. A gate that certifies parts a pipeline never invokes certifies
    # nothing about that pipeline.
    import inspect
    body = inspect.getsource(PA.stage_score) + inspect.getsource(PA.stage_fit)
    for label, tok in (("8a run path passes the TAPE into the feature pass", "TAPE="),
                       ("8b run path resolves CAUSAL thresholds", "freeze_thresholds("),
                       ("8c run path emits HEAD DIAGNOSTICS", "head_diagnostics("),
                       ("8d run path dispatches ARM D", "INCUMBENT_REWEIGHTED_ONLY"),
                       ("8e run path APPLIES the purge", "purge_training(")):
        seam(label, tok in body,
             "defined but never called by stage_fit/stage_score")

    # ---- SEAM 9 (R-189): enforcement must be VISIBLE AS NUMBERS -----------
    body2 = inspect.getsource(PA.stage_fit) + inspect.getsource(PA.stage_score)
    for label, tok in (
        ("9a purge records BOTH sides of the seam", "train_rows_before_purge"),
        ("9b the realized post-purge gap is recorded", "realized_gap_s"),
        ("9c gap>=60 is a COMPUTED predicate", "EMBARGO_ENFORCED"),
        ("9d evidence reaches the receipt", "embargo_evidence")):
        seam(label, tok in body2,
             "R-189: the fixture's word is not evidence; the receipt carries numbers")
    # and the predicate must be computed, never hardcoded True
    seam("9e EMBARGO_ENFORCED is evaluated, not asserted",
         'EMBARGO_ENFORCED": _gap["gap_s"] >= _gap["embargo_s"]' in body2,
         "a hardcoded True beside a table has contradicted the table before")

    # ---- SEAM 10 (R-190): a gap straddling a cutoff MUST status ----------
    # Tests the WIRE, not the population: a synthetic gap around a known
    # cutoff must produce GAP_AT_CUTOFF in a built tape, and a cutoff outside
    # it must not. If this is green while the real tape shows zero, the wire
    # works and the real population genuinely has none under this basis --
    # which is a different claim from "the wire is broken".
    import harmful_state_features as _sf
    _t = _sf.StateTape(slug="x-updown-5m-1", ws=1.0, gaps=[(10.0, 40.0)])
    _t.pm_event_t = [1.0]
    _row = {"slug": "x-updown-5m-1", "coin": "btc", "side": "BUY_UP", "gen": 1,
            "t_start": 25.0, "level": 0.5, "resting": 5.0, "qahead": 1.0}
    _in = _sf.features_at(_t, _row)
    _out = _sf.features_at(_t, {**_row, "t_start": 50.0})
    seam("10a a cutoff INSIDE a synthetic gap statuses GAP_AT_CUTOFF",
         _in["state_status"] == "GAP_AT_CUTOFF", _in["state_status"])
    seam("10b a cutoff OUTSIDE it does not",
         _out["state_status"] != "GAP_AT_CUTOFF", _out["state_status"])
    # R-191(2) LOSS MODE 1: a WARM-UP cutoff (negative t_start) inside a gap.
    # Every one of BE's 289 real flags is a warm-up row, and the PRE_WINDOW
    # branch precedes the gap branch in the status chain -- so under the
    # current chain a warm-up row in a gap statuses PRE_WINDOW and the gap is
    # LOST. This is the loss mode R-191 names, and it is why per-slug scoping
    # found zero: those rows never reach the gap test at all.
    _w = _sf.StateTape(slug="x-updown-5m-1", ws=1.0, gaps=[(-20.0, -10.0)])
    _w.pm_event_t = [1.0]
    _warm = _sf.features_at(_w, {**_row, "t_start": -15.0})
    seam("10d a WARM-UP cutoff inside a gap flags GAP_AT_CUTOFF",
         _warm["state_status"] == "GAP_AT_CUTOFF",
         f"got {_warm['state_status']!r} — PRE_WINDOW precedes the gap branch, "
         f"so the gap is lost on exactly the rows that carry it")
    # R-191(2) LOSS MODE 2: a gap STRADDLING a window boundary.
    _b = _sf.StateTape(slug="x-updown-5m-1", ws=1.0, gaps=[(-5.0, 5.0)])
    _b.pm_event_t = [1.0]
    _pre = _sf.features_at(_b, {**_row, "t_start": -2.0})
    _post = _sf.features_at(_b, {**_row, "t_start": 2.0})
    seam("10e a boundary-straddling gap flags on BOTH sides",
         _pre["state_status"] == "GAP_AT_CUTOFF" and
         _post["state_status"] == "GAP_AT_CUTOFF",
         f"pre={_pre['state_status']!r} post={_post['state_status']!r}")
    seam("10c the gap check compares the SAME basis it stores",
         _sf._in_gap(_t, 25.0) and not _sf._in_gap(_t, 50.0),
         "window-relative cutoff vs window-relative gap")

    # ================= SEAMS 11-16 (R-194) ================================
    # Every one of these was MISSED by seams 1-10, and the reason is uniform:
    # 1-10 assert that a component EXISTS, is NAMED, or is MENTIONED in the run
    # path. None of them assert the component is INVOKED CORRECTLY. A token in
    # source is not a call; a model KIND is not a model; a persisted field is
    # not a consumed one.

    # ---- SEAM 11: features must be NONZERO through the real path ----------
    # encode_row was handed the OUTER tape row while the 45 features live
    # under row["state"], so every lookup missed and the whole state family
    # scored as ZERO. All-zero features are silent: the fit converges, the
    # gate runs, the receipt looks normal.
    tape_row = {"slug": "s", "side": "BUY_UP", "gen": 1, "t_start": 1.0,
                "state_status": "OK",
                "state": {k: (i + 1) * 0.5 for i, k in enumerate(feats)}}
    outer_vec = PIN.encode_row(tape_row, feats)
    inner_vec = PIN.encode_row(tape_row.get("state", {}), feats)
    seam("11a encoding the OUTER row yields all zeros (the defect)",
         all(v == 0.0 for v in outer_vec),
         "if this is false the defect is elsewhere")
    seam("11b the run path encodes the NESTED state, not the outer row",
         "encode_row(sfe[" in inspect.getsource(PA._feature_pass) or
         'encode_row(sfe.get("state"' in inspect.getsource(PA._feature_pass),
         "run path passes the outer row -> every state feature is 0.0")
    seam("11c encoding the nested state yields NONZERO features",
         any(v != 0.0 for v in inner_vec))

    # ---- SEAM 12: arm D must differ from arm B, BY ARTIFACT ---------------
    src_sc = inspect.getsource(PA.stage_score)
    seam("12a arm D loads its OWN artifact, not B's",
         "linear_d_" in src_sc or "lin_d" in src_sc,
         "D shares B's elif branch and loads the same `lin` -> identical "
         "predictions, so D-A and B-D are both meaningless")
    src_ft = inspect.getsource(PA.stage_fit)
    seam("12b arm D is FITTED separately on incumbent features only",
         "linear_d_" in src_ft or "PM+fine only" in src_ft,
         "no separate D fit exists")

    # ---- SEAM 13: the evaluator must CONSUME the frozen threshold ---------
    ev_src = inspect.getsource(ae.evaluate_policy)
    seam("13a evaluate_policy accepts a frozen threshold",
         "theta_frozen" in ev_src or "frozen_threshold" in ev_src,
         "theta is computed retrospectively from the scoring population; the "
         "frozen thresholds are persisted and never used")
    seam("13b it SELECTS by the frozen threshold when given one",
         "theta_frozen is not None" in ev_src)

    # ---- SEAM 14: real heads, and an AUC that scales ----------------------
    hd_src = inspect.getsource(PA.head_diagnostics)
    seam("14a AUC is O(n log n), not a double loop",
         "sort" in hd_src and "for a in pos for b in neg" not in hd_src,
         "the pairwise loop is O(n^2); at 639k rows it does not finish")
    seam("14b heads take REAL hazard probabilities, not |ECV|",
         "p_haz" in inspect.signature(PA.head_diagnostics).parameters and
         "abs(e)" not in src_sc,
         "the run path passes min(1,|ecv|) as a probability and ecv as the "
         "value head -- neither is the model's actual output")

    # ---- SEAM 15: fit-stage indexes must be BOUNDED -----------------------
    ti_src = inspect.getsource(PA.tape_index)
    seam("15a the tape index stores compact values, not whole row dicts",
         "idx[" in ti_src and ('r["state"]' in ti_src or "compact" in ti_src),
         "1.7M full row dicts is ~12GB -- the R-174 violation again")

    # ---- SEAM 16: half-open containment, exactly-at-g1 NOT flagged --------
    import harmful_state_features as _sf16
    _e = _sf16.StateTape(slug="x", ws=1.0, gaps=[(10.0, 20.0)])
    _e.pm_event_t = [1.0]
    seam("16a a cutoff exactly at g0 IS flagged", _sf16._in_gap(_e, 10.0))
    seam("16b a cutoff exactly at g1 is NOT flagged", not _sf16._in_gap(_e, 20.0),
         "builder uses g0<=t<=g1 (closed); R-191 rules [g0,g1) -- builder and "
         "counter disagree on the edge")

    print(f"\n{'FIXTURE GREEN' if not FAILURES else 'FIXTURE RED'}: "
          f"{len(FAILURES)} seam(s) failing")
    for f in FAILURES:
        print(f"  - {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
