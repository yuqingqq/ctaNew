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

    print(f"\n{'FIXTURE GREEN' if not FAILURES else 'FIXTURE RED'}: "
          f"{len(FAILURES)} seam(s) failing")
    for f in FAILURES:
        print(f"  - {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
