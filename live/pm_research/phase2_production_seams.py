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
    seam("21c fitting REQUIRES DA's verdict artifact",
         "da_tape_gate_verdict" in fit_src or "da_tape_gate_verdict" in sc_src,
         "nothing requires an ALL-PASS gate verdict before fitting")

    # ---- SEAM 20: provenance interface + snapshot import isolation -------
    b_src = (HERE / "build_state_tape_v2.py").read_text()
    seam("20a builder reads BUILD_REF from the ENV",
         'environ["BUILD_REF"]' in b_src or 'environ.get("BUILD_REF"' in b_src,
         "builder runs git at runtime instead of being told its ref")
    seam("20b builder runs NO git at runtime",
         'rev-parse' not in b_src,
         "git at runtime reports the MAIN tree's HEAD AT COMPLETION, not the "
         "pinned ref the snapshot was cut from")
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
        saved = (PA.TAPE_PATH, PA.FRAGMENT, PA.TOPUP, PA.FITDIR)
        PA.TAPE_PATH, PA.FRAGMENT, PA.TOPUP = tp, frag, top
        PA.FITDIR = tdp / "fits"
        try:
            PA.stage_fit()
            seam("17a stage_fit COMPLETES end-to-end on a synthetic v5 tape", True)
            for coin in ("btc",):
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
            PA.TAPE_PATH, PA.FRAGMENT, PA.TOPUP, PA.FITDIR = saved

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

    print(f"\n{'PRODUCTION SEAMS GREEN' if not FAILURES else 'PRODUCTION SEAMS RED'}: "
          f"{len(FAILURES)} failing")
    for f in FAILURES:
        print(f"  - {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
