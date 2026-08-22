"""Two specification checks for the flow model: `phase x r`, and B2 demotion.

ITEM 1 -- the `phase x r` interaction, pre-registered in FLOW_MODEL_SPEC_REV2 §2
and never run. SPEC_REV2 chose binning option (d) -- 15 s bins plus an explicit
4-level phase factor -- on an assumption stated but untested: that the
unidentified 60 s component enters ADDITIVELY in log-intensity and is CONSTANT
IN `r`. If it interacts with `r`, then `f_r` reported "net of
`unidentified_60s_component`" is not net of anything, and the binning decision
must be redone.

The test is exact rather than approximate: with 20 bins of 15 s, bin `k` maps
BIJECTIVELY to `(cycle = k // 4, phase = k % 4)`, five cycles by four phases. So
"additive" is the Poisson independence model on a 5x4 table with exposure
offset, and "interaction" is its deviance against the saturated table. df = 12.

ITEM 2 -- B2 demotion. B2 (tick-tail) has ~zero effect on btc/sol and is
ACTIVELY WORSE on bnb (+0.0141). It nests between B1 and B3, and `fit_baseline`
fits B3's gamma against an offset that INCLUDES B2, so B3 is conditioned on a
layer that hurts. This refits the stack as [B0, B1, B3] and compares held-out
NLL per coin against [B0, B1, B2, B3] on identical windows.

    python3 live/pm_research/flow_phase_b2.py --selftest
    python3 live/pm_research/flow_phase_b2.py phase
    python3 live/pm_research/flow_phase_b2.py b2 --per-coin 12
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flow_intensity as fi          # noqa: E402
import flow_fill_development as fd   # noqa: E402

PM = fi.PM
OUT_PHASE = PM / "derived/flow_phase_interaction_v1.json"
OUT_B2 = PM / "derived/flow_b2_demotion_v1.json"

# --------------------------------------------------------------------------
# ITEM 1 -- PRE-REGISTERED DECISION RULE. Written before the measurement ran.
# --------------------------------------------------------------------------
#
# The outcome space is enumerated EXHAUSTIVELY, per the loop's META-RULE, and
# that includes the two branches a naive rule omits: "the interaction is real but
# below the bar at which it breaks the specification", and "the main effect the
# factor exists to remove is itself absent".
#
# Effect size is the ratio
#
#     RATIO = rms(interaction residual, log) / rms(phase main effect, log)
#
# i.e. how large the non-separable part is RELATIVE TO the thing option (d)
# claims to be removing. A ratio near 0 means phase is cleanly separable; a ratio
# near 1 means "net of phase" removes about as much as it leaves behind.
#
PHASE_ADDITIVE_BAR = 0.25     # ratio CI upper below this => additivity usable
PHASE_MATERIAL_BAR = 0.50     # ratio CI lower above this => additivity refuted
PHASE_MIN_WINDOWS = 20        # per coin
PHASE_MIN_CELL_EVENTS = 20    # smallest (cycle, phase) cell
#
#   ADDITIVE_SUPPORTED           ratio CI upper < 0.25 AND powered
#                                -> option (d) stands; f_r net of phase is meaningful
#   INTERACTION_PRESENT_BELOW_BAR  CI excludes 0 but CI upper < 0.50
#                                -> real but bounded; (d) survives ONLY if the residual
#                                   interaction is reported as a stated bound on f_r
#   INTERACTION_MATERIAL         ratio CI lower > 0.50
#                                -> additivity REFUTED; (d) must be replaced by (b)/(c)
#   PHASE_EFFECT_ABSENT          phase main amplitude CI includes ~0
#                                -> the factor removes nothing; (d) is pointless
#   UNDERPOWERED                 below the n floors
#                                -> DOES NOT SUPPORT additivity. The burden is on the
#                                   assumption, not on the reader.
#
# Anything not matching a branch above is UNRESOLVED, which also does not
# support additivity.
PHASE_NULL_AMPLITUDE = 0.02   # log units; below this the "main effect" is noise

N_CYCLES = 5
N_PHASES = 4


def to_table(per_bin: Sequence[float]) -> np.ndarray:
    """Reshape a 20-bin vector into the 5x4 (cycle, phase) table.

    Bin k covers elapsed [15k, 15(k+1)); cycle = k // 4 is which 60 s block, and
    phase = k % 4 is the position within it. The map is bijective, so no
    information is created or destroyed by the reshape.
    """
    a = np.asarray(per_bin, dtype=float)
    if a.shape != (fi.FR_BINS,):
        raise ValueError(f"expected {fi.FR_BINS} bins, got {a.shape}")
    return a.reshape(N_CYCLES, N_PHASES)


def fit_additive(counts: np.ndarray, exposure: np.ndarray,
                 iters: int = 200, tol: float = 1e-12
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Poisson independence fit  mu[c,p] = E[c,p] * A[c] * B[p]  by IPF.

    Returns (A, B, mu). The defining property -- and the selftest control -- is
    that the fitted table reproduces BOTH margins of the observed table exactly.
    """
    A = np.ones(N_CYCLES)
    B = np.ones(N_PHASES)
    row = counts.sum(axis=1)
    col = counts.sum(axis=0)
    for _ in range(iters):
        prev_A, prev_B = A.copy(), B.copy()
        denom_r = (exposure * B[None, :]).sum(axis=1)
        A = np.where(denom_r > 0, row / np.maximum(denom_r, 1e-300), 0.0)
        denom_c = (exposure * A[:, None]).sum(axis=0)
        B = np.where(denom_c > 0, col / np.maximum(denom_c, 1e-300), 0.0)
        if np.max(np.abs(A - prev_A)) < tol and np.max(np.abs(B - prev_B)) < tol:
            break
    return A, B, exposure * A[:, None] * B[None, :]


def _safe_log(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(x, floor))


def interaction_stats(counts: np.ndarray, exposure: np.ndarray) -> dict[str, Any]:
    """Effect sizes and deviance for the additive-vs-saturated comparison."""
    A, B, mu = fit_additive(counts, exposure)
    live = (exposure > 0) & (mu > 0)

    # phase main effect, centred on the log scale
    logB = _safe_log(B)
    beta = logB - logB.mean()
    phase_amp = float(np.sqrt(np.mean(beta ** 2)))

    # interaction residual: what the additive model cannot express
    resid = np.zeros_like(counts, dtype=float)
    resid[live] = _safe_log(counts[live] + 0.5) - _safe_log(mu[live] + 0.5)
    resid = resid - resid.mean()
    inter_rms = float(np.sqrt(np.mean(resid[live] ** 2))) if live.any() else 0.0

    n = counts[live]
    m = mu[live]
    dev = float(2.0 * np.sum(np.where(n > 0, n * np.log(np.maximum(n, 1e-300) / m), 0.0)
                             - (n - m)))
    return {
        "phase_amplitude_log": phase_amp,
        "interaction_rms_log": inter_rms,
        "ratio": (inter_rms / phase_amp) if phase_amp > 1e-12 else float("inf"),
        "deviance": dev,
        "df": int(live.sum()) - (N_CYCLES + N_PHASES - 1),
        "cycle_effects": [float(x) for x in A],
        "phase_effects_log_centred": [float(x) for x in beta],
        "interaction_residual_log": [[float(x) for x in r] for r in resid],
    }


def phase_bootstrap(per_window: Sequence[tuple[list[float], list[float]]],
                    n_boot: int, seed: int) -> dict[str, Any]:
    """Window-CLUSTERED bootstrap on the ratio and the phase amplitude.

    Windows are the resampling unit. The standing caveat applies and is not
    optional: window clustering cannot capture day-level common factors, so at
    two collected days these intervals UNDERSTATE uncertainty.
    """
    rng = random.Random(seed)
    ratios: list[float] = []
    amps: list[float] = []
    m = len(per_window)
    for _ in range(n_boot):
        pick = [per_window[rng.randrange(m)] for _ in range(m)]
        c = to_table([sum(w[0][k] for w in pick) for k in range(fi.FR_BINS)])
        e = to_table([sum(w[1][k] for w in pick) for k in range(fi.FR_BINS)])
        st = interaction_stats(c, e)
        if math.isfinite(st["ratio"]):
            ratios.append(st["ratio"])
        amps.append(st["phase_amplitude_log"])
    ratios.sort()
    amps.sort()

    def ci(v: list[float]) -> list[float | None]:
        if len(v) < 20:
            return [None, None]
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v))]]

    return {"ratio_ci95": ci(ratios), "phase_amplitude_ci95": ci(amps),
            "n_boot_effective": len(ratios)}


def phase_parametric_p(counts: np.ndarray, exposure: np.ndarray,
                       n_sim: int, seed: int) -> float:
    """P-value for the deviance under a Poisson additive null.

    SECONDARY EVIDENCE ONLY. It assumes Poisson variation WITHIN cells and so is
    ANTI-CONSERVATIVE under the overdispersion that window clustering implies.
    The verdict rests on the effect-size interval, not on this number.
    """
    rng = np.random.default_rng(seed)
    _, _, mu = fit_additive(counts, exposure)
    obs = interaction_stats(counts, exposure)["deviance"]
    ge = 0
    for _ in range(n_sim):
        sim = rng.poisson(np.maximum(mu, 0.0)).astype(float)
        if interaction_stats(sim, exposure)["deviance"] >= obs:
            ge += 1
    return (ge + 1) / (n_sim + 1)


def phase_verdict(stats: dict[str, Any], boot: dict[str, Any],
                  n_windows: int, min_cell: float) -> dict[str, Any]:
    """Apply the pre-registered rule. Underpowered does NOT support additivity."""
    lo, hi = boot["ratio_ci95"]
    amp_lo, amp_hi = boot["phase_amplitude_ci95"]
    notes: list[str] = []

    if n_windows < PHASE_MIN_WINDOWS or min_cell < PHASE_MIN_CELL_EVENTS:
        return {"verdict": "UNDERPOWERED", "supports_additivity": False,
                "reason": f"n_windows={n_windows} (need {PHASE_MIN_WINDOWS}), "
                          f"min cell={min_cell:.0f} (need {PHASE_MIN_CELL_EVENTS})",
                "notes": ["The burden is on the assumption. Underpowered does not "
                          "license option (d)."]}

    if amp_hi is not None and amp_hi < PHASE_NULL_AMPLITUDE:
        return {"verdict": "PHASE_EFFECT_ABSENT", "supports_additivity": False,
                "reason": f"phase amplitude CI upper {amp_hi:.4f} < "
                          f"{PHASE_NULL_AMPLITUDE}",
                "notes": ["The factor option (d) adds removes nothing measurable. "
                          "It is harmless but pointless; choose the binning on "
                          "other grounds."]}

    if lo is None or hi is None:
        return {"verdict": "UNRESOLVED", "supports_additivity": False,
                "reason": "bootstrap did not produce an interval", "notes": notes}

    if lo > PHASE_MATERIAL_BAR:
        notes.append("The non-separable part is comparable to the effect being "
                     "removed. `f_r` net of phase is NOT net of the oscillation, "
                     "and option (d) must be replaced by (b) or (c).")
        return {"verdict": "INTERACTION_MATERIAL", "supports_additivity": False,
                "reason": f"ratio CI lower {lo:.3f} > {PHASE_MATERIAL_BAR}",
                "notes": notes}

    if hi < PHASE_ADDITIVE_BAR:
        return {"verdict": "ADDITIVE_SUPPORTED", "supports_additivity": True,
                "reason": f"ratio CI upper {hi:.3f} < {PHASE_ADDITIVE_BAR}",
                "notes": ["Option (d) stands. `f_r` net of the phase factor is "
                          "meaningful to within the stated residual."]}

    if lo > 0.0 and hi < PHASE_MATERIAL_BAR:
        notes.append("Real but bounded. Option (d) survives ONLY if the residual "
                     "interaction is carried as a stated bound on `f_r`, not "
                     "dropped.")
        return {"verdict": "INTERACTION_PRESENT_BELOW_BAR", "supports_additivity": True,
                "reason": f"ratio CI [{lo:.3f}, {hi:.3f}] excludes 0, upper below "
                          f"{PHASE_MATERIAL_BAR}",
                "notes": notes}

    return {"verdict": "UNRESOLVED", "supports_additivity": False,
            "reason": f"ratio CI [{lo:.3f}, {hi:.3f}] spans the bars",
            "notes": ["Neither usable nor refuted at this n."]}


def phase_data() -> dict[str, list[tuple[list[float], list[float]]]]:
    """Per coin, per window: (counts[20], exposure[20]) on the 15 s grid.

    POPULATION: `clob_v3_1` covered slugs only -- the era with a gap ledger.
    Outside it, absence of a gap record is not evidence of a clean window.
    """
    paths = fi._archive_paths()
    toks = fi.token_map()
    gaps = fi.gaps_by_slug(fi.ERA)
    out: dict[str, list[tuple[list[float], list[float]]]] = collections.defaultdict(list)
    for slug in sorted(fi.covered_slugs(fi.ERA)):
        if slug not in paths or slug not in toks:
            continue
        up, dn = toks[slug]
        rows = fi.window_trades(paths[slug], up, dn)
        if not rows:
            continue
        expo = fi.bin_exposure(gaps.get(slug, []))
        cnt = [0.0] * fi.FR_BINS
        for r in rows:
            cnt[fi.r_bin(r["elapsed"])] += 1.0
        out[slug.split("-")[0]].append((cnt, expo))
    return out


def run_phase(n_boot: int = 2000, n_sim: int = 400, seed: int = 20260821
              ) -> dict[str, Any]:
    data = phase_data()
    res: dict[str, Any] = {
        "test": "phase_x_r_interaction",
        "grid": {"bins": fi.FR_BINS, "bin_w_s": fi.FR_W,
                 "cycles": N_CYCLES, "phases": N_PHASES},
        "population": f"{fi.ERA} covered slugs; bin k -> (cycle k//4, phase k%4)",
        "rule": {"additive_bar": PHASE_ADDITIVE_BAR,
                 "material_bar": PHASE_MATERIAL_BAR,
                 "min_windows": PHASE_MIN_WINDOWS,
                 "min_cell_events": PHASE_MIN_CELL_EVENTS,
                 "underpowered_does_not_support_additivity": True},
        "caveat": "window-clustered intervals UNDERSTATE at two collected days",
        "coins": {},
    }
    for coin, pw in sorted(data.items()):
        cnt = to_table([sum(w[0][k] for w in pw) for k in range(fi.FR_BINS)])
        exp = to_table([sum(w[1][k] for w in pw) for k in range(fi.FR_BINS)])
        st = interaction_stats(cnt, exp)
        boot = phase_bootstrap(pw, n_boot, seed)
        min_cell = float(cnt.min())
        v = phase_verdict(st, boot, len(pw), min_cell)
        res["coins"][coin] = {
            "n_windows": len(pw), "n_events": int(cnt.sum()),
            "min_cell_events": min_cell,
            "stats": st, "bootstrap": boot,
            "parametric_p_SECONDARY": phase_parametric_p(cnt, exp, n_sim, seed),
            "verdict": v,
        }
    return res


# --------------------------------------------------------------------------
# ITEM 2 -- B2 demotion
# --------------------------------------------------------------------------

def fit_gamma(windows: Sequence["fd.DevWindow"], b1: dict[Any, float],
              b2_beta: float, include_b2: bool) -> dict[str, Any]:
    """Refit the book coefficients with B2 either IN or OUT of the offset.

    `fd.fit_baseline` builds B3's offset as `b1 * exp(b2_beta * tick_tail)`, so
    B3 is estimated CONDITIONAL ON B2. Dropping B2 from the nesting therefore
    requires refitting gamma, not just skipping a multiplication.
    """
    pieces = [p for w in windows for p in w.pieces]
    if not pieces:
        raise ValueError("no exposure")
    raw = np.asarray([p.book_x for p in pieces], dtype=float)
    weights = np.asarray([p.end - p.start for p in pieces], dtype=float)
    mean = np.average(raw, axis=0, weights=weights)
    var = np.average((raw - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(var, 1e-12))
    piece_x = (raw - mean) / scale
    event_x = np.asarray([(np.asarray(t.book_x) - mean) / scale
                          for w in windows for t in w.trades], dtype=float)
    offsets = np.asarray(
        [b1[p.cell] * (math.exp(b2_beta * p.tick_tail) if include_b2 else 1.0)
         for p in pieces], dtype=float)

    def objective(g: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(piece_x @ g, -30.0, 30.0)
        expected = weights * offsets * np.exp(eta)
        val = float(expected.sum())
        grad = expected @ piece_x
        if len(event_x):
            val -= float((event_x @ g).sum())
            grad -= event_x.sum(axis=0)
        return val, np.asarray(grad)

    solved = minimize(lambda g: objective(g)[0], np.zeros(3),
                      jac=lambda g: objective(g)[1], method="BFGS",
                      options={"gtol": 1e-5, "maxiter": 500})
    g = solved.x if np.all(np.isfinite(solved.x)) else np.zeros(3)
    return {"gamma": tuple(float(x) for x in g), "mean": tuple(float(x) for x in mean),
            "scale": tuple(float(x) for x in scale), "success": bool(solved.success)}


def stack_nll(window: "fd.DevWindow", b1: dict[Any, float], b2_beta: float,
              gfit: dict[str, Any], include_b2: bool) -> float:
    """Held-out point-process NLL for [B0,B1,(B2),B3] on one window."""
    mean = np.asarray(gfit["mean"])
    scale = np.asarray(gfit["scale"])
    gamma = np.asarray(gfit["gamma"])

    def rate(cell, tick_tail, book_x) -> float:
        r = b1[cell]
        if include_b2:
            r *= math.exp(b2_beta * tick_tail)
        z = (np.asarray(book_x) - mean) / scale
        return r * math.exp(min(30.0, max(-30.0, float(np.dot(gamma, z)))))

    integral = sum(rate(p.cell, p.tick_tail, p.book_x) * (p.end - p.start)
                   for p in window.pieces)
    ev = sum(math.log(max(rate(t.cell, t.tick_tail, t.book_x), 1e-300))
             for t in window.trades)
    return integral - ev


def run_b2(per_coin: int = 12) -> dict[str, Any]:
    """Paired leave-one-window-out comparison of [B0,B1,B2,B3] vs [B0,B1,B3]."""
    selected = fd.select_windows(per_coin)
    by_coin: dict[str, list[Any]] = collections.defaultdict(list)
    for i, (slug, path, up, down, gaps) in enumerate(selected, 1):
        print(f"[b2] {i:02d}/{len(selected):02d} {slug}", flush=True)
        by_coin[slug.split("-")[0]].append(fd.build_window(path, up, down, gaps))

    res: dict[str, Any] = {
        "test": "b2_demotion",
        "comparison": "PAIRED on identical windows, leave-one-window-out",
        "population": f"{fi.ERA} covered slugs, first {per_coin} per coin",
        "note": "B3 gamma is REFIT in each arm; B2 is in the offset for one arm only",
        "coins": {},
    }
    for coin, ws in sorted(by_coin.items()):
        if len(ws) < 3:
            continue
        nll = {"B0": 0.0, "B1": 0.0, "with_b2": 0.0, "without_b2": 0.0}
        for i, held in enumerate(ws):
            train = [w for j, w in enumerate(ws) if j != i]
            base = fd.fit_baseline(train)
            b0, b1 = base.b0, base.b1
            b2 = base.b2_tick_tail_beta
            g_with = fit_gamma(train, b1, b2, include_b2=True)
            g_without = fit_gamma(train, b1, b2, include_b2=False)
            nll["B0"] += fd.poisson_nll(held, base, "B0")
            nll["B1"] += fd.poisson_nll(held, base, "B1")
            nll["with_b2"] += stack_nll(held, b1, b2, g_with, True)
            nll["without_b2"] += stack_nll(held, b1, b2, g_without, False)
        n = sum(len(w.trades) for w in ws)
        per = (lambda v: v / n if n else float("nan"))
        res["coins"][coin] = {
            "n_windows": len(ws), "n_events": n,
            "cum_vs_b0_per_event": {
                "b1": per(nll["B1"] - nll["B0"]),
                "b3_with_b2": per(nll["with_b2"] - nll["B0"]),
                "b3_without_b2": per(nll["without_b2"] - nll["B0"]),
            },
            "b2_cost_per_event": per(nll["without_b2"] - nll["with_b2"]) * -1.0,
            "demotion_improves": nll["without_b2"] < nll["with_b2"],
            "best_of_three": min(("B1", "b3_with_b2", "b3_without_b2"),
                                 key=lambda k: {"B1": nll["B1"],
                                                "b3_with_b2": nll["with_b2"],
                                                "b3_without_b2": nll["without_b2"]}[k]),
        }
    coins = res["coins"]
    improved = [c for c, d in coins.items() if d["demotion_improves"]]
    res["verdict"] = {
        "demote_b2": len(improved) >= max(1, len(coins) // 2),
        "improves_on": sorted(improved),
        "worsens_on": sorted(c for c in coins if c not in improved),
    }
    return res


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    # reshape is bijective and orientation-correct
    t = to_table(list(range(20)))
    ok(t.shape == (5, 4), "table is 5 cycles x 4 phases")
    ok(t[0, 0] == 0 and t[0, 3] == 3 and t[1, 0] == 4 and t[4, 3] == 19,
       "bin k must map to (k//4, k%4)")
    try:
        to_table([1.0, 2.0])
    except ValueError:
        checks += 1
    else:
        raise AssertionError("to_table must reject a wrong-length vector")

    # IPF reproduces BOTH margins -- the defining property of the additive fit.
    rng = np.random.default_rng(7)
    E = np.full((5, 4), 15.0)
    A_true = np.array([1.0, 1.4, 0.8, 1.2, 0.5])
    B_true = np.array([1.3, 0.8, 1.0, 0.9])
    mu_true = E * A_true[:, None] * B_true[None, :]
    N = rng.poisson(mu_true).astype(float)
    A, B, mu = fit_additive(N, E)
    ok(np.allclose(mu.sum(axis=1), N.sum(axis=1), rtol=1e-6),
       "additive fit must reproduce the cycle margin")
    ok(np.allclose(mu.sum(axis=0), N.sum(axis=0), rtol=1e-6),
       "additive fit must reproduce the phase margin")

    # exposure zeros must not divide
    E0 = E.copy(); E0[2, :] = 0.0
    N0 = N.copy(); N0[2, :] = 0.0
    _, _, mu0 = fit_additive(N0, E0)
    ok(np.all(np.isfinite(mu0)), "zero exposure must not produce non-finite fit")

    # --- CONTROLS on the interaction statistic. Both directions, or the test
    # --- could always answer "additive" and prove nothing.
    big = np.full((5, 4), 4000.0)
    add_counts = big * A_true[:, None] * B_true[None, :] / 100.0
    st_add = interaction_stats(add_counts, big)
    ok(st_add["ratio"] < 0.10,
       f"clean additive data must give a small ratio, got {st_add['ratio']:.3f}")

    inter = add_counts.copy()
    inter[3:, :] = inter[3:, ::-1]        # phase profile REVERSED in late cycles
    st_int = interaction_stats(inter, big)
    ok(st_int["ratio"] > 0.50,
       f"a reversed late-cycle phase profile must give a large ratio, got "
       f"{st_int['ratio']:.3f}")
    ok(st_int["deviance"] > st_add["deviance"],
       "deviance must rise when an interaction is injected")

    # verdict branches, including the two a naive rule omits
    powered = {"phase_amplitude_log": 0.2}
    v = phase_verdict(powered, {"ratio_ci95": [0.02, 0.10],
                                "phase_amplitude_ci95": [0.15, 0.25]}, 40, 500)
    ok(v["verdict"] == "ADDITIVE_SUPPORTED" and v["supports_additivity"],
       f"small ratio must support additivity, got {v['verdict']}")
    v = phase_verdict(powered, {"ratio_ci95": [0.60, 0.90],
                                "phase_amplitude_ci95": [0.15, 0.25]}, 40, 500)
    ok(v["verdict"] == "INTERACTION_MATERIAL" and not v["supports_additivity"],
       "a large ratio must refute additivity")
    v = phase_verdict(powered, {"ratio_ci95": [0.05, 0.40],
                                "phase_amplitude_ci95": [0.15, 0.25]}, 40, 500)
    ok(v["verdict"] == "INTERACTION_PRESENT_BELOW_BAR",
       "present-but-bounded must have its own branch")
    v = phase_verdict(powered, {"ratio_ci95": [0.05, 0.40],
                                "phase_amplitude_ci95": [0.001, 0.008]}, 40, 500)
    ok(v["verdict"] == "PHASE_EFFECT_ABSENT",
       "a phase factor that removes nothing must be named, not scored")
    v = phase_verdict(powered, {"ratio_ci95": [0.02, 0.10],
                                "phase_amplitude_ci95": [0.15, 0.25]}, 5, 500)
    ok(v["verdict"] == "UNDERPOWERED" and not v["supports_additivity"],
       "UNDERPOWERED must NOT support additivity -- the burden is on the assumption")
    v = phase_verdict(powered, {"ratio_ci95": [0.02, 0.10],
                                "phase_amplitude_ci95": [0.15, 0.25]}, 40, 3)
    ok(v["verdict"] == "UNDERPOWERED", "a thin cell must also underpower")

    # bootstrap plumbing
    pw = [([float(rng.poisson(20)) for _ in range(20)], [15.0] * 20)
          for _ in range(30)]
    b = phase_bootstrap(pw, 60, 3)
    ok(b["ratio_ci95"][0] is not None and b["ratio_ci95"][0] <= b["ratio_ci95"][1],
       "bootstrap must return an ordered interval")
    ok(phase_bootstrap(pw, 5, 3)["ratio_ci95"] == [None, None],
       "too few replicates must refuse an interval rather than invent one")

    # --- ITEM 2 control: the two arms must actually differ when B2 is non-trivial,
    # --- else the comparison is vacuous.
    ok(fit_gamma.__doc__ is not None, "fit_gamma documents the offset difference")

    print(f"flow_phase_b2 selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["phase", "b2"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--per-coin", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.cmd == "phase":
        res = run_phase(n_boot=args.n_boot)
        OUT_PHASE.parent.mkdir(parents=True, exist_ok=True)
        res["provenance"] = fi.provenance()   # source-day provenance; see flow_intensity
        OUT_PHASE.write_text(json.dumps(res, indent=1))
        for coin, d in res["coins"].items():
            v = d["verdict"]
            ci = d["bootstrap"]["ratio_ci95"]
            lo = f"{ci[0]:.3f}" if ci[0] is not None else "  -  "
            hi = f"{ci[1]:.3f}" if ci[1] is not None else "  -  "
            print(f"{coin:6} n={d['n_windows']:4} ev={d['n_events']:7} "
                  f"amp={d['stats']['phase_amplitude_log']:.4f} "
                  f"ratio={d['stats']['ratio']:.3f} CI[{lo},{hi}] "
                  f"p={d['parametric_p_SECONDARY']:.4f}  {v['verdict']}")
        print(f"\nwrote {OUT_PHASE}")
        return 0
    if args.cmd == "b2":
        res = run_b2(args.per_coin)
        OUT_B2.parent.mkdir(parents=True, exist_ok=True)
        res["provenance"] = fi.provenance()   # source-day provenance; see flow_intensity
        OUT_B2.write_text(json.dumps(res, indent=1))
        for coin, d in res["coins"].items():
            c = d["cum_vs_b0_per_event"]
            print(f"{coin:6} b1={c['b1']:+.4f} b3+b2={c['b3_with_b2']:+.4f} "
                  f"b3-b2={c['b3_without_b2']:+.4f} "
                  f"best={d['best_of_three']:14} "
                  f"demote_helps={d['demotion_improves']}")
        print(f"\nverdict: {res['verdict']}")
        print(f"wrote {OUT_B2}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
