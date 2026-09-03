#!/usr/bin/env python3
"""R-217 INCREMENT NULL — is the candidate-over-incumbent increment chance?

WHY THIS EXISTS. The Phase-2 gate tests each arm against a matched RANDOM
canceller. It never tested arm-vs-arm. So "beats_random_max_on_NET: True" on
every arm says nothing about whether PLUS_PRED_STATE_V1's +2,785c over
INCUMBENT_REWEIGHTED_ONLY is more than four other candidates' worth of noise --
and the increment is the quantity a freeze decision turns on. Beating random is
not beating the incumbent.

DECLARED IN THE REGISTER BEFORE ANY COMPUTATION (coordinator R-217, 72a1764).
The design below is transcribed from that declaration and is not to be widened
here; anything this file computes beyond it is a separate, later question.

  cells       coin {btc,eth} x candidate {PLUS_PRED_STATE_V1, LGBM_PINNED}
              x budget {5%,10%,15%}  =  12
  statistic   TOTAL net-cents increment vs INCUMBENT_REWEIGHTED_ONLY over the
              identical action population (same rows; each arm makes its own
              cancel decisions on them)
  null        WINDOW-LEVEL SIGN-FLIP permutation of per-window paired
              increment sums
  n_perm      >= 1000
  p           two-sided, per cell
  reading     12 cells read JOINTLY; multiplicity named, not absorbed
  G           computed and STATED; no intervals claimed
  scope       reads committed artifacts; touches no pipeline file

THE UNIT IS WEAKER THAN THE RULED ONE, AND THAT IS DISCLOSED, NOT HIDDEN.
CLAUDE.md rule 8 makes the UTC DAY the cluster unit. G is expected to be 0
complete days here, which is exactly why no interval is claimed. The window is
used as the permutation unit because it is the finest unit that is plausibly
exchangeable; windows within a day are NOT independent, so a p-value from this
test is optimistic. It is reported as evidence, never as a significance
certificate.

REPLICATION IS PROVEN, NOT ASSUMED. This file re-derives each arm's net from
the committed fit artifacts. If that re-derivation is not IDENTICAL to the
committed receipt's net_cents for every cell, the run REFUSES: a null computed
on a slightly different statistic than the one reported would be worse than no
null at all.
"""
from __future__ import annotations

import json, math, os, random, sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

import phase2_arms as PA
import phase2_declaration as D
import harmful_action_eval as ae
import harmful_hazard_model as hm

DERIVED = PA.DERIVED
RECEIPT = DERIVED / "phase2_four_arm_v2.json"
OUT = DERIVED / "phase2_increment_null_v1.json"

CANDIDATES = ("PLUS_PRED_STATE_V1", "LGBM_PINNED")
BASELINE = "INCUMBENT_REWEIGHTED_ONLY"
N_PERM = 2000                      # declared >= 1000
PERM_SEED = 20260827
RECON_TOL = 1e-6                   # cents; identity, not approximation


def _self_sha16() -> str:
    """Content identity of THIS analysis file -- measured, not a passed label."""
    import hashlib
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()[:16]


def assert_reconciles(got: float, want: float, label: str,
                      tol: float = RECON_TOL) -> float:
    """Return |got - want|, refusing above tol. CALLABLE, so a probe can drive it.

    DA F2: the check was inline in main() and could not be exercised, so the
    guard's own behaviour was asserted in prose rather than demonstrated."""
    delta = abs(got - want)
    if delta > tol:
        raise RuntimeError(
            f"REFUSED: re-derived net for {label} is {got!r} but the committed "
            f"receipt says {want!r} (delta {got - want:+.6f}c, tol {tol}). A "
            f"null computed on a different statistic than the one reported is "
            f"worse than no null.")
    return delta


def per_window_net(rows, scores, theta):
    """EXACT replication of evaluate_policy's causal net, tallied per window.

    Every step mirrors the evaluator: generations keyed (slug, side, gen), rows
    within a generation ordered by t_start, a generation cancels iff its MAX
    score reaches theta, and it acts ONCE at its FIRST crossing row, valued by
    that row's latency-aware preventable value."""
    L = str(D.TARGET_LATENCY_MS)
    gens: dict = {}
    for i, r in enumerate(rows):
        gens.setdefault((r.get("slug"), r["side"], r["gen"]), []).append(i)
    for k in gens:
        gens[k].sort(key=lambda i: rows[i]["t_start"])
    by_window: dict = {}
    total = 0.0
    n_cancelled = 0
    for k, idxs in gens.items():
        if max(scores[i] for i in idxs) < theta:
            continue
        cross = next(i for i in idxs if scores[i] >= theta)
        r = rows[cross]
        v = (r["latency"][L]["preventable_value_cents"]
             if r.get("any_fill_ahead") and "latency" in r else 0.0)
        by_window[k[0]] = by_window.get(k[0], 0.0) + v
        total += v
        n_cancelled += 1
    return by_window, total, n_cancelled


def ordered_windows(a: dict, b: dict) -> list:
    """The permutation's window order, DETERMINISTIC and callable.

    R-234. Order must be a function of the DATA, never of set iteration: a
    fixed RNG seed applied to a process-dependent order reproduces nothing.
    """
    return sorted(set(a) | set(b))


def sign_flip_p(inc_by_window: dict, n_perm: int = N_PERM, seed: int = PERM_SEED):
    """Two-sided p from a window-level sign-flip permutation.

    H0: the per-window paired increments are symmetric about zero. Each
    permutation flips each window's sign independently and re-sums."""
    # BEM-R8: SORT AT CONSUMPTION. This function was order-dependent -- the
    # same content in two insertion orders gave 0.0719 and 0.0938 on the real
    # published increments -- and its one live caller was correct only because
    # it pinned the order first via `ordered_windows`. That is the caller-side
    # reliance R-234 exists to forbid: the next caller inherits the defect.
    # `phase2_iter011.sign_flip_null` already sorts here for the same reason.
    vals = [float(inc_by_window[k]) for k in sorted(inc_by_window)]
    obs = math.fsum(vals)
    rng = random.Random(seed)
    ge = 0
    null_max = -float("inf")
    null_abs = []
    for _ in range(n_perm):
        s = math.fsum(v if rng.getrandbits(1) else -v for v in vals)
        null_abs.append(abs(s))
        null_max = max(null_max, s)
        if abs(s) >= abs(obs):
            ge += 1
    null_abs.sort()
    return {
        "observed_increment_cents": obs,
        "n_windows": len(vals),
        "n_windows_positive": sum(1 for v in vals if v > 0),
        "n_windows_negative": sum(1 for v in vals if v < 0),
        "n_windows_zero": sum(1 for v in vals if v == 0),
        "n_perm": n_perm,
        "perm_seed": seed,
        # (ge + 1) / (n + 1): the permutation p can never be 0 -- the observed
        # arrangement is itself one of the arrangements under H0.
        "p_two_sided": (ge + 1) / (n_perm + 1),
        "unit_order": "SORTED KEYS — pinned at consumption (R-234, BEM-R8)",
        "null_abs_p95": null_abs[int(0.95 * len(null_abs))] if null_abs else None,
        "null_max_signed": null_max,
    }


def complete_utc_days(rows) -> dict:
    """G, computed, with the definition stated in the artifact.

    A UTC date D is COMPLETE iff the population's time span covers the whole of
    D: min(t) <= D 00:00:00 and max(t) >= D+1 00:00:00. Partial days are listed
    with their action counts so a reader can see what the span actually holds."""
    import datetime as dt
    ts = [r["t0"] + r["t_start"] for r in rows]
    lo, hi = min(ts), max(ts)
    by_date: dict = {}
    for t in ts:
        d = dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat()
        by_date[d] = by_date.get(d, 0) + 1
    complete = []
    for d in sorted(by_date):
        d0 = dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp()
        if lo <= d0 and hi >= d0 + 86400:
            complete.append(d)
    return {
        "definition": ("a UTC date is COMPLETE iff the population span covers "
                       "the whole of it: min(t) <= 00:00:00 and max(t) >= the "
                       "next 00:00:00"),
        "G_complete_utc_days": len(complete),
        "complete_days": complete,
        "dates_present": sorted(by_date),
        "rows_by_date": {k: by_date[k] for k in sorted(by_date)},
        "span_utc": [
            dt.datetime.fromtimestamp(lo, dt.timezone.utc).isoformat(),
            dt.datetime.fromtimestamp(hi, dt.timezone.utc).isoformat()],
        "span_hours": (hi - lo) / 3600.0,
        "intervals_claimed": False,
        "why": ("CLAUDE.md rule 8: below G=5 complete days, a point estimate "
                "and no interval. G is STATED here rather than inferred from "
                "the absence of an interval."),
    }


def selftest() -> int:
    """Falsifiers (rule 15): each instrument must fire on a known-bad input."""
    fails = []

    def ok(c, label):
        print(f"  {'PASS' if c else 'FAIL'}  {label}")
        if not c:
            fails.append(label)

    # sign_flip_p: a large one-sided effect must be improbable; pure noise must not
    strong = {f"w{i}": 100.0 for i in range(40)}
    r1 = sign_flip_p(strong, n_perm=500, seed=1)
    ok(r1["p_two_sided"] < 0.01, "sign-flip flags a 40-window all-positive effect")
    alt = {f"w{i}": (100.0 if i % 2 else -100.0) for i in range(40)}
    r2 = sign_flip_p(alt, n_perm=500, seed=1)
    ok(r2["p_two_sided"] > 0.2, "sign-flip does NOT flag a balanced null")
    ok(r1["p_two_sided"] > 0, "p is never exactly 0 ((ge+1)/(n+1))")

    # ---- R-234 determinism falsifiers -----------------------------------
    # KNOWN-BAD (DA's reversed-insertion demo): the permutation IS order
    # dependent, which is why the order must be pinned. If this ever stops
    # differing, the falsifier has gone blind and the repair proves nothing.
    _d1 = {"w1": 100.0, "w2": -50.0, "w3": 25.0, "w4": -10.0, "w5": 60.0, "w6": -80.0}
    _d2 = {k: _d1[k] for k in reversed(list(_d1))}
    _r1, _r2 = sign_flip_p(_d1, n_perm=500, seed=1), sign_flip_p(_d2, n_perm=500, seed=1)
    ok(_r1["observed_increment_cents"] == _r2["observed_increment_cents"],
       "known-bad: the observed increment is order-INdependent (a sum)")
    # BEM-R8: this check USED to assert that the p IS order-dependent -- a
    # control whose subject was the hole, kept green by the very defect it
    # described. The repair now lives at consumption (`sorted(inc_by_window)`),
    # so the assertion is inverted: the p must be order-INVARIANT, and a
    # regression that removed the sort would turn this red instead of green.
    ok(_r1["p_two_sided"] == _r2["p_two_sided"],
       f"BEM-R8: the permutation p is now order-INVARIANT at consumption "
       f"({_r1['p_two_sided']}) -- a caller no longer has to know to pin the "
       f"order first, which is the reliance R-234 forbids")
    _shuf = {k: _d1[k] for k in sorted(_d1, key=lambda x: _d1[x])}
    ok(sign_flip_p(_shuf, n_perm=500, seed=1)["p_two_sided"]
       == _r1["p_two_sided"],
       "BEM-R8: and a THIRD insertion order agrees too, so the equality above "
       "is invariance and not a coincidence of reversing")
    ok(_r1["unit_order"].startswith("SORTED KEYS"),
       "BEM-R8: the artifact SAYS the order it consumed in, so a reader need "
       "not infer it from two p-values")
    # THE REPAIR: ordered_windows is a function of the DATA, not of insertion
    ok(ordered_windows(_d1, {}) == ordered_windows(_d2, {}) == sorted(_d1),
       "ordered_windows is identical under any insertion order")
    ok(sign_flip_p({k: _d1[k] for k in ordered_windows(_d1, {})}, n_perm=500, seed=1)
       ["p_two_sided"]
       == sign_flip_p({k: _d2[k] for k in ordered_windows(_d2, {})}, n_perm=500, seed=1)
       ["p_two_sided"],
       "after ordering, p is IDENTICAL from either insertion order")
    # END-TO-END: two interpreters under DIFFERENT PYTHONHASHSEED must agree
    import subprocess as _sp, os as _os, sys as _sys
    _prog = ("import sys;sys.path.insert(0,%r)\n"
             "import phase2_increment_null as M\n"
             "d={('w%%02d'%%i):(i*7%%13)-6.0 for i in range(40)}\n"
             "w=M.ordered_windows(d,{})\n"
             "print(M.sign_flip_p({k:d[k] for k in w},n_perm=400,seed=5)['p_two_sided'])"
             % str(Path(__file__).resolve().parent))
    _outs = []
    for _hs in ("0", "12345"):
        _e = dict(_os.environ, PYTHONHASHSEED=_hs)
        _outs.append(_sp.run([_sys.executable, "-c", _prog], capture_output=True,
                             text=True, env=_e).stdout.strip())
    ok(_outs[0] and _outs[0] == _outs[1],
       f"two interpreters under DIFFERENT PYTHONHASHSEED agree exactly "
       f"({_outs[0]} == {_outs[1]}) -- the repair holds across processes, "
       f"which the seed alone never did")

    # per_window_net: the evaluator's semantics, on a hand-built case
    L = str(D.TARGET_LATENCY_MS)
    def row(slug, gen, t, v, fill=True):
        return {"slug": slug, "side": "BUY_UP", "gen": gen, "t_start": t,
                "t0": 1787650200.0, "any_fill_ahead": fill,
                "latency": {L: {"preventable_value_cents": v,
                                "preventable_shares": 1.0 if fill else 0.0,
                                "stale_shares": 0.0}}}
    rows = [row("A", 1, 0.0, 5.0), row("A", 1, 1.0, 99.0),
            row("B", 1, 0.0, 7.0)]
    scores = [0.9, 0.95, 0.1]
    bw, tot, nc = per_window_net(rows, scores, theta=0.5)
    ok(bw == {"A": 5.0} and tot == 5.0 and nc == 1,
       "generation acts ONCE at its FIRST crossing, later rows inert")
    bw2, tot2, _ = per_window_net(rows, scores, theta=99.0)
    ok(bw2 == {} and tot2 == 0.0, "a threshold above every gmax cancels NOTHING")
    rows_nofill = [row("A", 1, 0.0, 5.0, fill=False)]
    _, tot3, nc3 = per_window_net(rows_nofill, [0.9], theta=0.5)
    ok(tot3 == 0.0 and nc3 == 1,
       "a cancelled generation with no fill ahead is valued 0, not skipped")
    # ordering: rows out of t_start order must still cross at the EARLIEST row
    rows_rev = [row("A", 1, 5.0, 50.0), row("A", 1, 0.0, 3.0)]
    bwr, totr, _ = per_window_net(rows_rev, [0.9, 0.9], theta=0.5)
    ok(totr == 3.0, "rows are ordered by t_start, not by list position")

    # assert_reconciles: must PASS inside tol and REFUSE outside it
    ok(assert_reconciles(1.0, 1.0, "x") == 0.0, "reconciliation passes on identity")
    ok(assert_reconciles(1.0, 1.0 + RECON_TOL / 2, "x") > 0,
       "reconciliation returns the MEASURED delta, not a boolean")
    try:
        assert_reconciles(1.0, 2.0, "known-bad")
        ok(False, "reconciliation REFUSES a mismatch")
    except RuntimeError as e:
        ok("known-bad" in str(e), "reconciliation REFUSES a mismatch")

    print(f"\n{'INCREMENT-NULL SELFTEST GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing")
    return 1 if fails else 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    if selftest():
        raise SystemExit("REFUSED: selftest RED; no null is computed from an "
                         "instrument that cannot prove it fires.")
    import harmful_fast_compute as fc
    import lightgbm as lgb
    import numpy as np

    PA.assert_modules_under_root()
    PA.pin_data_root()
    PA.assert_tape_is_v5()
    _v = PA.assert_gate_passed()
    PA.assert_verdict_subject_is(PA.TAPE_PATH, _v)
    _mani = PA.assert_fit_complete_and_matching()
    receipt = json.loads(RECEIPT.read_text())
    if receipt.get("protocol") != "PHASE2_FOUR_ARM_V2":
        raise RuntimeError(f"REFUSED: {RECEIPT.name} is not the four-arm receipt.")

    print("  indexing score split...", flush=True)
    TP = PA.tape_index("score")
    SC = PA._feature_pass(PA.TOPUP, "topup", TAPE=TP)
    del TP

    frozen = json.loads(PA.FROZEN.read_text())
    cells = {}
    populations = {}
    for coin in ("btc", "eth"):
        if coin not in SC or not SC[coin]["kept"]:
            continue
        sc = SC[coin]
        lin = json.loads((PA.FITDIR / f"linear_{coin}.json").read_text())
        srows = [hm.keptrow(r) for r in sc["kept"]]
        n_sc = len(srows)
        populations[coin] = complete_utc_days(srows)
        populations[coin]["score_rows"] = n_sc
        populations[coin]["score_actions"] = len(
            {(r["slug"], r["side"], r["gen"]) for r in srows})
        populations[coin]["score_windows"] = len({r["slug"] for r in srows})

        def _raw(i):
            return sc["PM"][i] + sc["FN"][i] + sc["ST"][i]

        def ecv_for(arm):
            """Byte-for-byte the same construction stage_score used."""
            if arm == "INCUMBENT_REWEIGHTED_ONLY":
                a = json.loads((PA.FITDIR / f"linear_d_{coin}.json").read_text())
                mu, sd, W, WM = (a["norm_mu"], a["norm_sd"],
                                 a["hazard_weights"], a["value_weights"])
                out = []
                for j in range(n_sc):
                    raw = sc["PM"][j] + sc["FN"][j]
                    x = [1.0] + [(raw[i] - mu[i]) / sd[i] for i in range(len(mu))]
                    ph = fc.fast_predict_p(W, x)
                    vh = float(sum(p * q for p, q in zip(WM, x))) if WM else 0.0
                    out.append(ph * vh)
                return out, a.get("causal_thresholds")
            if arm == "PLUS_PRED_STATE_V1":
                mu, sd = lin["norm_mu"], lin["norm_sd"]
                W, WM = lin["hazard_weights"], lin["value_weights"]
                out = []
                for j in range(n_sc):
                    raw = _raw(j)
                    x = [1.0] + [(raw[i] - mu[i]) / sd[i] for i in range(len(mu))]
                    ph = fc.fast_predict_p(W, x)
                    vh = float(sum(p * q for p, q in zip(WM, x))) if WM else 0.0
                    out.append(ph * vh)
                return out, lin.get("causal_thresholds")
            if arm == "LGBM_PINNED":
                mu, sd = lin["norm_mu"], lin["norm_sd"]
                hb = lgb.Booster(model_file=str(PA.FITDIR / f"lgbm_haz_{coin}.txt"))
                vf = PA.FITDIR / f"lgbm_val_{coin}.txt"
                vb = lgb.Booster(model_file=str(vf)) if vf.exists() else None
                CH, out = 50_000, []
                for lo in range(0, n_sc, CH):
                    hi = min(lo + CH, n_sc)
                    S = np.empty((hi - lo, len(mu) + 1), dtype=np.float64)
                    S[:, 0] = 1.0
                    for j in range(lo, hi):
                        raw = _raw(j)
                        S[j - lo, 1:] = [(raw[i] - mu[i]) / sd[i]
                                         for i in range(len(mu))]
                    p = hb.predict(S)
                    v = vb.predict(S) if vb is not None else np.zeros(hi - lo)
                    out.extend((p * v).tolist())
                    del S
                _tf = PA.FITDIR / f"lgbm_thresholds_{coin}.json"
                return out, json.loads(_tf.read_text())
            raise RuntimeError(f"unhandled arm {arm}")

        base_ecv, base_thr = ecv_for(BASELINE)
        base_by_budget = {}
        for b in D.BUDGETS:
            key = f"{int(b * 100)}%"
            bw, tot, nc = per_window_net(srows, base_ecv, float(base_thr[key]))
            base_by_budget[key] = (bw, tot, nc)
        del base_ecv

        for cand in CANDIDATES:
            c_ecv, c_thr = ecv_for(cand)
            for b in D.BUDGETS:
                key = f"{int(b * 100)}%"
                cbw, ctot, cnc = per_window_net(srows, c_ecv, float(c_thr[key]))
                bbw, btot, bnc = base_by_budget[key]
                # RECONCILE against the committed receipt before anything else.
                # DA F1: this recorded a hardcoded True beside the numbers it
                # claimed to describe. The MEASURED worst-case delta is
                # recorded instead, so the claim is the evidence.
                _deltas = [assert_reconciles(
                    got, receipt["arms"][coin][arm]["gate"]["budgets"][key]["net_cents"],
                    f"{coin}/{arm}@{key}")
                    for arm, got in ((cand, ctot), (BASELINE, btot))]
                # R-234 DETERMINISM REPAIR (DA blocker-7 finding). This was
                # `for w in set(cbw) | set(bbw)`. sign_flip_p assigns signs in
                # LIST ORDER, and a set of window-key strings iterates in an
                # order that varies per process (string hashing is randomised
                # and PYTHONHASHSEED was unpinned). So PERM_SEED pinned the RNG
                # but NOT the data the RNG was applied to: every run was an
                # INDEPENDENT DRAW, not a reproduction. Measured across two
                # runs: increments bit-identical, 11 of 12 p-values moved, up
                # to 0.0245 -- the scale of Monte-Carlo error at n_perm=2000.
                # Sorting makes the draw a function of the data alone.
                wins = ordered_windows(cbw, bbw)
                inc = {w: cbw.get(w, 0.0) - bbw.get(w, 0.0) for w in wins}
                res = sign_flip_p(inc)
                res.update({
                    "coin": coin, "candidate": cand, "baseline": BASELINE,
                    "budget": key,
                    "candidate_net_cents": ctot, "baseline_net_cents": btot,
                    "candidate_cancellations": cnc,
                    "baseline_cancellations": bnc,
                    "max_abs_delta_cents": max(_deltas),
                    "reconciliation_tolerance_cents": RECON_TOL,
                    "reconciles_with_receipt": max(_deltas) <= RECON_TOL,
                    "threshold_mode": "CAUSAL_FROZEN_FROM_TRAIN"})
                cells[f"{coin}/{cand}/{key}"] = res
                print(f"  {coin}/{cand}@{key}: inc {res['observed_increment_cents']:+9.1f}c  "
                      f"p={res['p_two_sided']:.4f}  "
                      f"windows +{res['n_windows_positive']}/-{res['n_windows_negative']}",
                      flush=True)
            del c_ecv
        del base_by_budget, srows, SC[coin]["PM"], SC[coin]["FN"], SC[coin]["ST"]

    ps = sorted(c["p_two_sided"] for c in cells.values())
    m = len(ps)
    # Holm-Bonferroni across the 12 cells, since they are READ JOINTLY.
    holm = {}
    for i, (k, c) in enumerate(sorted(cells.items(), key=lambda kv: kv[1]["p_two_sided"])):
        holm[k] = min(1.0, c["p_two_sided"] * (m - i))
    for i, k in enumerate(sorted(holm, key=lambda k: cells[k]["p_two_sided"])):
        if i:
            prev = sorted(holm, key=lambda k: cells[k]["p_two_sided"])[i - 1]
            holm[k] = max(holm[k], holm[prev])       # enforce monotonicity

    out = {
        "artifact": "phase2_increment_null_v1",
        "authority": "coordinator R-217 (register 72a1764) — design DECLARED "
                     "before any computation; transcribed here unwidened",
        "question": "is the candidate-over-incumbent net increment "
                    "distinguishable from chance at the window level?",
        "why_needed": "the Phase-2 gate tests each arm against a matched RANDOM "
                      "canceller and never tested arm-vs-arm; beating random is "
                      "not beating the incumbent",
        "statistic": "total net-cents increment vs INCUMBENT_REWEIGHTED_ONLY "
                     "over the identical action population",
        "null": "window-level sign-flip permutation of per-window paired "
                "increment sums",
        "n_perm": N_PERM, "perm_seed": PERM_SEED,
        "determinism": {
            "ruling": "R-234 (8da983e) -- determinism repair, not re-selection",
            "window_order": "sorted(set(candidate)|set(baseline)) via "
                            "ordered_windows(); previously raw set iteration",
            "pythonhashseed": __import__("os").environ.get("PYTHONHASHSEED"),
            "why": "sign_flip_p assigns signs in LIST ORDER, so a fixed "
                   "PERM_SEED over a process-dependent order is an "
                   "INDEPENDENT DRAW each run, not a reproduction",
            "prior_draws_are_provenance": (
                "two independent MC estimates of the same true p exist: "
                "e7caaeb (R-217) and the 2026-08-28 re-binding preserved at "
                "phase2_increment_null_v1.SUPERSEDED_BY_v2.json. Increments "
                "were bit-identical across both; 11 of 12 p-values moved, max "
                "0.0245, consistent with MC error at n_perm=2000 (SE~0.0112)"),
            "acceptance": "PRE-COMMITTED: this run's p-values are the numbers "
                          "of record whatever they are; a Holm survival flip "
                          "IS the result, reported never reconciled. One run."},
        "n_cells": m, "cells_read": "JOINTLY",
        "multiplicity": {
            "n_cells": m,
            "correction": "Holm-Bonferroni across all cells",
            "note": "the 12 cells are one family read together; a single "
                    "uncorrected cell is not a result"},
        "cluster_unit_disclosure": {
            "ruled_unit": "UTC day (CLAUDE.md rule 8)",
            "unit_used": "window (slug)",
            "weaker_than_ruled": True,
            "why": "G is 0 complete UTC days, so the ruled unit has no "
                   "replicates. The window is the finest plausibly-exchangeable "
                   "unit available; windows within a day are NOT independent, "
                   "so these p-values are OPTIMISTIC and are evidence, not a "
                   "significance certificate."},
        "reconciliation": {
            "every_cell_reconciled_to_receipt": all(
                c["reconciles_with_receipt"] for c in cells.values()),
            "max_abs_delta_cents_over_all_cells": max(
                (c["max_abs_delta_cents"] for c in cells.values()), default=None),
            "tolerance_cents": RECON_TOL,
            "receipt": RECEIPT.name,
            "why": "the null must be computed on the SAME statistic the "
                   "receipt reports, proven rather than assumed"},
        "provenance": {
            "tape_sha256_prefix": _mani.get("tape_sha256_prefix"),
            "fit_code_ref": _mani.get("fit_code_ref"),
            "verdict_sha256_prefix": _mani.get("verdict_sha256_prefix"),
            "gate_code_sha256_prefix": _mani.get("gate_code_sha256_prefix"),
            "receipt_commit_note": "receipt committed at 288df3f",
            # the fit ref names the FIT; it does not name the analysis code
            # that produced these numbers. Both are recorded.
            "analysis_code_ref": os.environ.get("ANALYSIS_CODE_REF", "").strip() or None,
            "analysis_code_sha256_prefix": _self_sha16()},
        "populations": populations,
        "cells": cells,
        "holm_adjusted_p": holm,
    }
    with OUT.open("w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.flush(); os.fsync(fh.fileno())
    print(f"\nWROTE {OUT.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
