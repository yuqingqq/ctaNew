"""Non-uniform `f_r` grid, and whether the terminal mechanism is still confounded.

TASK 1 -- the replacement binning. SPEC_REV2 option (d), 15 s bins plus an
additive 4-level phase factor, is REFUTED: `INTERACTION_MATERIAL` on all seven
coins, ratio 1.685 (btc) to 4.486 (hype) against a 0.50 bar, every ratio above
1.0. The interaction concentrates in the terminal minute, so `f_r` has real
structure finer than 60 s there and a cycle-level factor cannot express it.
Option (b), uniform 60 s bins, buries that collapse in one bin.

The replacement is a NON-UNIFORM grid, and each half is chosen against measured
structure:

  BODY  elapsed [0, 240) -> 4 bins of 60 s.
        A 60 s bin spans EXACTLY ONE PERIOD of the unidentified 60 s component,
        so it absorbs that component by construction rather than by assumption --
        which is the only honest treatment of a term whose SOURCE is
        unidentifiable (window phase and wall-clock minute phase are perfectly
        collinear here). The body r-profile is flat-to-mildly-declining, so 60 s
        resolution discards nothing measurable: the within-minute range in the
        body is only 1.19-1.41x (btc), against 6.49x in the terminal.

  TERMINAL elapsed [240, 300) -> 12 bins of 5 s.
        The collapse is 5.9x (eth) to 9.1x (hype) across a single minute and is
        `f_r`'s largest feature. 5 s gives twelve points to resolve its shape,
        and the thinnest coin still holds a few hundred arrivals per bin.

TASK 2 -- the terminal mechanism. FLOW_MODEL_STATE §2 records it as
unidentifiable because the settlement TWAP is 60 s, the oscillation is 60 s, and
every window start is congruent to 0 mod 60. But the two hypotheses predict
DIFFERENT SHAPES, which the phase x r result did not test:

  TWAP lock-in is CONTINUOUS -- the settled fraction of a 60 s trailing mean
  rises smoothly from r=60, so activity should decay monotonically through the
  final minute, and the effect should be specific to that minute.

  A minute-boundary artefact is PERIODIC -- it should appear with comparable
  amplitude at EVERY minute boundary, not only the last.

WHAT THIS TEST CAN AND CANNOT DO. It can refute "a uniform minute-boundary
artefact explains the terminal collapse". It CANNOT positively establish TWAP
lock-in: a non-stationary artefact, or a distinct real effect that happens to sit
in the last minute, would look the same. `TWAP_FAVOURED` therefore means the
uniform-artefact explanation is refuted and TWAP remains the standing candidate,
NOT that TWAP is established. That distinction is in the verdict text.

    python3 live/pm_research/flow_grid_terminal.py --selftest
    python3 live/pm_research/flow_grid_terminal.py grid
    python3 live/pm_research/flow_grid_terminal.py terminal
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flow_intensity as fi  # noqa: E402

PM = fi.PM
OUT_GRID = PM / "derived/flow_grid_nonuniform_v1.json"
OUT_TERM = PM / "derived/flow_terminal_mechanism_v1.json"
OUT_MD = Path(__file__).with_name("FLOW_GRID_TERMINAL_RESULTS.md")

# --------------------------------------------------------------------------
# the non-uniform grid
# --------------------------------------------------------------------------

BODY_END_S = 240.0          # elapsed; r = 60 s remaining
BODY_W = 60.0
TERMINAL_W = 5.0

MODEL_EDGES: tuple[float, ...] = tuple(
    [i * BODY_W for i in range(int(BODY_END_S / BODY_W))]
    + [BODY_END_S + j * TERMINAL_W
       for j in range(int((fi.WINDOW_S - BODY_END_S) / TERMINAL_W) + 1)]
)
N_MODEL_BINS = len(MODEL_EDGES) - 1

# Task 2 needs body and terminal minutes compared at IDENTICAL resolution, so it
# uses a uniform fine grid rather than the model grid.
FINE_W = 5.0
FINE_BINS = int(fi.WINDOW_S / FINE_W)      # 60
FINE_PER_MINUTE = int(60.0 / FINE_W)       # 12
N_MINUTES = 5

# --------------------------------------------------------------------------
# TASK 2 -- PRE-REGISTERED DECISION RULE. Written before the measurement ran.
# --------------------------------------------------------------------------
#
# Statistic A -- AMPLITUDE RATIO. Within-minute log range (max-min of the
# exposure-corrected log profile) for the terminal minute, divided by the MEAN
# of the same quantity over the four body minutes. A periodic artefact of
# constant amplitude gives ~1.
#
# Statistic B -- MONOTONICITY. Spearman rho of the terminal minute's 12 fine
# bins against r. TWAP lock-in is a smooth decay; a boundary artefact is a step.
#
# Statistic C -- ONSET. Log jump across r=60 (last body fine bin -> first
# terminal fine bin), reported for interpretation. NOT a verdict input: it was
# inspected in the phase receipt before this rule was written, so using it to
# decide would be circular.
#
TERM_MATERIAL_BAR = 3.0     # amplitude ratio at/above this => materially weaker elsewhere
TERM_COMPARABLE_BAR = 1.5   # at/below this => comparable at earlier boundaries
TERM_MONOTONE_RHO = -0.80   # at/below this => monotone decline
TERM_MIN_BIN_EVENTS = 100   # smallest terminal fine bin, per coin
VERDICT_COINS = ("btc", "eth")   # FLOW_MODEL_PROTOCOL_V4 verdict_coins
#
#   TWAP_FAVOURED       amplitude-ratio CI LOWER >= 3.0 and rho <= -0.80,
#                       on btc AND eth. Means: the uniform-artefact explanation
#                       is REFUTED. Does NOT establish TWAP.
#   ARTEFACT_FAVOURED   amplitude-ratio CI UPPER <= 1.5 on btc AND eth.
#   STILL_CONFOUNDED    anything in between, or the two coins disagree, or the
#                       terminal decline is not monotone.
#   UNRESOLVED          below the event floor. Does NOT support either mechanism
#                       and is reported alongside STILL_CONFOUNDED consequences.
#
# Underpowered defaults to the confounded reading. The attractive answer here is
# TWAP -- it is the mechanism a reader who knows the settlement rule reaches for
# immediately -- so the burden sits on it.


def model_bin(elapsed_s: float) -> int:
    """Index into the non-uniform grid. Half-open [lo, hi), last bin closed."""
    if not (0.0 <= elapsed_s <= fi.WINDOW_S):
        raise ValueError(f"elapsed {elapsed_s} outside window")
    if elapsed_s >= MODEL_EDGES[-1]:
        return N_MODEL_BINS - 1
    lo, hi = 0, N_MODEL_BINS
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if elapsed_s >= MODEL_EDGES[mid]:
            lo = mid
        else:
            hi = mid
    return lo


def model_bin_label(k: int) -> str:
    a, b = MODEL_EDGES[k], MODEL_EDGES[k + 1]
    return f"r{fi.WINDOW_S - b:.0f}-{fi.WINDOW_S - a:.0f}"


def fine_bin(elapsed_s: float) -> int:
    if not (0.0 <= elapsed_s <= fi.WINDOW_S):
        raise ValueError(f"elapsed {elapsed_s} outside window")
    return min(int(elapsed_s / FINE_W), FINE_BINS - 1)


def edge_exposure(gaps: Sequence[tuple[float, float]],
                  edges: Sequence[float]) -> list[float]:
    """Observed seconds per bin for an arbitrary edge set.

    Gaps may straddle boundaries, so overlap is computed per bin rather than by
    assigning a gap to one bin.
    """
    out = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        lost = sum(fi.overlap(lo, hi, g0, g1) for g0, g1 in gaps)
        out.append(max(0.0, (hi - lo) - lost))
    return out


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------

def scan(era_only: bool = True) -> dict[str, list[dict[str, Any]]]:
    """One pass over the immutable archives, accumulating BOTH grids."""
    paths = fi._archive_paths()
    toks = fi.token_map()
    gaps = fi.gaps_by_slug(fi.ERA)
    cov = fi.covered_slugs(fi.ERA)
    slugs = sorted(cov if era_only else set(paths))
    slugs = [s for s in slugs if s in paths and s in toks]

    fine_edges = [i * FINE_W for i in range(FINE_BINS + 1)]
    by_coin: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for slug in slugs:
        up, dn = toks[slug]
        rows = fi.window_trades(paths[slug], up, dn)
        if not rows:
            continue
        g = gaps.get(slug, []) if era_only else []
        rec = {
            "slug": slug,
            "m_cnt": [0.0] * N_MODEL_BINS, "m_cnt_ex": [0.0] * N_MODEL_BINS,
            "m_notl": [0.0] * N_MODEL_BINS, "m_notl_ex": [0.0] * N_MODEL_BINS,
            "m_expo": edge_exposure(g, MODEL_EDGES),
            "f_cnt": [0.0] * FINE_BINS, "f_notl": [0.0] * FINE_BINS,
            "f_expo": edge_exposure(g, fine_edges),
        }
        for r in rows:
            k = model_bin(r["elapsed"])
            j = fine_bin(r["elapsed"])
            rec["m_cnt"][k] += 1.0
            rec["m_notl"][k] += r["notional"]
            rec["f_cnt"][j] += 1.0
            rec["f_notl"][j] += r["notional"]
            if not r["micro"]:
                rec["m_cnt_ex"][k] += 1.0
                rec["m_notl_ex"][k] += r["notional"]
        by_coin[slug.split("-")[0]].append(rec)
    return by_coin


# --------------------------------------------------------------------------
# TASK 1 -- f_r on the non-uniform grid
# --------------------------------------------------------------------------

MICRO_PRIMARY_NOTIONAL = 0.35   # R-DUAL: above this micro share, count is not evidence


def grid(n_boot: int = 2000, seed: int = 20260822) -> dict[str, Any]:
    by_coin = scan()
    res: dict[str, Any] = {
        "grid": {"edges_elapsed_s": list(MODEL_EDGES), "n_bins": N_MODEL_BINS,
                 "body_w_s": BODY_W, "terminal_w_s": TERMINAL_W,
                 "body_end_elapsed_s": BODY_END_S},
        "population": f"{fi.ERA} covered slugs, days {'/'.join(fi.DAYS)}; "
                      f"denominator is OBSERVED exposure per bin (gap-corrected), "
                      f"numerator is folded taker arrivals in the same bin",
        "caveat": "window-clustered intervals cannot capture day-level common "
                  "factors; at two collected days they UNDERSTATE uncertainty",
        "coins": {},
    }
    for coin, ws in sorted(by_coin.items()):
        n_all = sum(sum(w["m_cnt"]) for w in ws)
        n_ex = sum(sum(w["m_cnt_ex"]) for w in ws)
        micro_share = (n_all - n_ex) / n_all if n_all else float("nan")
        d: dict[str, Any] = {
            "n_windows": len(ws),
            "n_trades": int(n_all),
            "micro_event_share": micro_share,
            "primary_weighting": ("NOTIONAL_ONLY"
                                  if micro_share > MICRO_PRIMARY_NOTIONAL
                                  else "EITHER"),
            "exposure_s": sum(sum(w["m_expo"]) for w in ws),
        }
        for key, num in (("count", "m_cnt"), ("count_ex_micro", "m_cnt_ex"),
                         ("notional", "m_notl"), ("notional_ex_micro", "m_notl_ex")):
            pw = [(w[num], w["m_expo"]) for w in ws]
            prof = fi.profile_ratio(pw)
            d[key] = {"profile": prof, "shape_ratio": fi.shape_ratio(prof)}
            if key in ("count", "notional"):
                d[key]["ci"] = fi.cluster_bootstrap(pw, n_boot, seed)
        # terminal collapse measured ON THE MODEL GRID
        for key in ("count", "notional"):
            p = [x for x in d[key]["profile"][-12:] if math.isfinite(x) and x > 0]
            d[key]["terminal_collapse_ratio"] = (max(p) / min(p)) if len(p) >= 2 else None
        res["coins"][coin] = d
    return res


# --------------------------------------------------------------------------
# TASK 2 -- terminal mechanism
# --------------------------------------------------------------------------

def _minute_profiles(cnt: Sequence[float], expo: Sequence[float]
                     ) -> list[list[float]]:
    """Exposure-corrected log profile for each minute, centred within minute."""
    out = []
    for m in range(N_MINUTES):
        vals = []
        for j in range(FINE_PER_MINUTE):
            k = m * FINE_PER_MINUTE + j
            rate = (cnt[k] / expo[k]) if expo[k] > 0 else float("nan")
            vals.append(math.log(rate) if (math.isfinite(rate) and rate > 0)
                        else float("nan"))
        finite = [v for v in vals if math.isfinite(v)]
        mean = sum(finite) / len(finite) if finite else 0.0
        out.append([(v - mean) if math.isfinite(v) else float("nan") for v in vals])
    return out


def _log_range(prof: Sequence[float]) -> float:
    v = [x for x in prof if math.isfinite(x)]
    return (max(v) - min(v)) if len(v) >= 2 else float("nan")


def spearman_rho(y: Sequence[float]) -> float:
    """Rank correlation of y against its index. Ties averaged."""
    v = [(i, x) for i, x in enumerate(y) if math.isfinite(x)]
    n = len(v)
    if n < 3:
        return float("nan")
    order = sorted(range(n), key=lambda i: v[i][1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]][1] == v[order[i]][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    xs = list(range(1, n + 1))
    mx, my = sum(xs) / n, sum(ranks) / n
    num = sum((xs[i] - mx) * (ranks[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((r - my) ** 2 for r in ranks))
    return (num / (dx * dy)) if dx > 0 and dy > 0 else float("nan")


def terminal_stats(cnt: Sequence[float], expo: Sequence[float]) -> dict[str, Any]:
    profs = _minute_profiles(cnt, expo)
    body = [_log_range(p) for p in profs[:-1]]
    term = _log_range(profs[-1])
    body_ok = [b for b in body if math.isfinite(b)]
    body_mean = sum(body_ok) / len(body_ok) if body_ok else float("nan")
    ratio = (term / body_mean) if (math.isfinite(term) and body_mean > 1e-12) else float("nan")
    # onset: last body fine bin -> first terminal fine bin, on the RAW log rate
    def lograte(k: int) -> float:
        return (math.log(cnt[k] / expo[k])
                if expo[k] > 0 and cnt[k] > 0 else float("nan"))
    a, b = lograte(N_MINUTES * FINE_PER_MINUTE - FINE_PER_MINUTE - 1), lograte(
        (N_MINUTES - 1) * FINE_PER_MINUTE)
    return {
        "body_minute_log_ranges": body,
        "body_mean_log_range": body_mean,
        "terminal_log_range": term,
        "amplitude_ratio": ratio,
        "terminal_spearman_rho": spearman_rho(profs[-1]),
        "body_spearman_rho": [spearman_rho(p) for p in profs[:-1]],
        "onset_log_jump_at_r60_INTERPRETIVE": (b - a) if
            (math.isfinite(a) and math.isfinite(b)) else float("nan"),
        "terminal_profile_log": profs[-1],
    }


def terminal_bootstrap(pw: Sequence[tuple[list[float], list[float]]],
                       n_boot: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    ratios, rhos = [], []
    m = len(pw)
    for _ in range(n_boot):
        pick = [pw[rng.randrange(m)] for _ in range(m)]
        c = [sum(w[0][k] for w in pick) for k in range(FINE_BINS)]
        e = [sum(w[1][k] for w in pick) for k in range(FINE_BINS)]
        st = terminal_stats(c, e)
        if math.isfinite(st["amplitude_ratio"]):
            ratios.append(st["amplitude_ratio"])
        if math.isfinite(st["terminal_spearman_rho"]):
            rhos.append(st["terminal_spearman_rho"])

    def ci(v: list[float]) -> list[float | None]:
        if len(v) < 20:
            return [None, None]
        v = sorted(v)
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v))]]

    return {"amplitude_ratio_ci95": ci(ratios), "spearman_rho_ci95": ci(rhos),
            "n_boot_effective": len(ratios)}


def terminal_verdict(per_coin: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-registered rule on the verdict coins only."""
    notes: list[str] = []
    have = [c for c in VERDICT_COINS if c in per_coin]
    if not have:
        return {"verdict": "UNRESOLVED", "supports_twap": False,
                "reason": "no verdict coin present", "notes": notes}

    under = [c for c in have
             if per_coin[c]["min_terminal_bin_events"] < TERM_MIN_BIN_EVENTS]
    if under:
        return {"verdict": "UNRESOLVED", "supports_twap": False,
                "reason": f"terminal bins below {TERM_MIN_BIN_EVENTS} events: {under}",
                "notes": ["Underpowered does NOT support either mechanism; the "
                          "confounded reading stands."]}

    twap, art = [], []
    for c in have:
        lo, hi = per_coin[c]["bootstrap"]["amplitude_ratio_ci95"]
        rho = per_coin[c]["stats"]["terminal_spearman_rho"]
        twap.append(lo is not None and lo >= TERM_MATERIAL_BAR
                    and math.isfinite(rho) and rho <= TERM_MONOTONE_RHO)
        art.append(hi is not None and hi <= TERM_COMPARABLE_BAR)

    if all(twap):
        notes.append("REFUTES the uniform minute-boundary artefact as an "
                     "explanation of the terminal collapse. It does NOT establish "
                     "TWAP lock-in: a non-stationary artefact, or a distinct real "
                     "effect confined to the last minute, predicts the same shape.")
        return {"verdict": "TWAP_FAVOURED", "supports_twap": True,
                "reason": "amplitude-ratio lower bound >= "
                          f"{TERM_MATERIAL_BAR} and monotone on {list(have)}",
                "notes": notes}
    if all(art):
        return {"verdict": "ARTEFACT_FAVOURED", "supports_twap": False,
                "reason": "amplitude-ratio upper bound <= "
                          f"{TERM_COMPARABLE_BAR} on {list(have)}", "notes": notes}
    return {"verdict": "STILL_CONFOUNDED", "supports_twap": False,
            "reason": "verdict coins disagree, the ratio sits between the bars, "
                      "or the terminal decline is not monotone",
            "notes": ["The confounded reading in FLOW_MODEL_STATE section 2 stands."]}


def terminal(n_boot: int = 2000, seed: int = 20260822) -> dict[str, Any]:
    by_coin = scan()
    per_coin: dict[str, dict[str, Any]] = {}
    for coin, ws in sorted(by_coin.items()):
        cnt = [sum(w["f_cnt"][k] for w in ws) for k in range(FINE_BINS)]
        expo = [sum(w["f_expo"][k] for w in ws) for k in range(FINE_BINS)]
        pw = [(w["f_cnt"], w["f_expo"]) for w in ws]
        term_events = cnt[(N_MINUTES - 1) * FINE_PER_MINUTE:]
        per_coin[coin] = {
            "n_windows": len(ws),
            "n_trades": int(sum(cnt)),
            "min_terminal_bin_events": min(term_events) if term_events else 0.0,
            "stats": terminal_stats(cnt, expo),
            "bootstrap": terminal_bootstrap(pw, n_boot, seed),
        }
    return {
        "test": "terminal mechanism: TWAP lock-in vs uniform minute-boundary artefact",
        "grid": {"fine_w_s": FINE_W, "bins": FINE_BINS,
                 "per_minute": FINE_PER_MINUTE},
        "population": f"{fi.ERA} covered slugs, days {'/'.join(fi.DAYS)}; "
                      f"amplitude ratio denominator is the MEAN within-minute log "
                      f"range over the FOUR BODY MINUTES of the same coin",
        "rule": {"material_bar": TERM_MATERIAL_BAR,
                 "comparable_bar": TERM_COMPARABLE_BAR,
                 "monotone_rho": TERM_MONOTONE_RHO,
                 "min_bin_events": TERM_MIN_BIN_EVENTS,
                 "verdict_coins": list(VERDICT_COINS),
                 "can_refute": "uniform minute-boundary artefact",
                 "cannot_establish": "TWAP lock-in"},
        "caveat": "window-clustered intervals UNDERSTATE uncertainty at two days",
        "coins": per_coin,
        "verdict": terminal_verdict(per_coin),
    }


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def _expect(label: str, exc: type[BaseException], fn) -> None:
    try:
        fn()
    except exc:
        return
    except BaseException as other:  # noqa: BLE001
        raise AssertionError(f"{label}: expected {exc.__name__}, got {other!r}") from other
    raise AssertionError(f"{label}: expected {exc.__name__}, nothing raised")


def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    # --- grid geometry
    ok(N_MODEL_BINS == 16, f"4 body + 12 terminal, got {N_MODEL_BINS}")
    ok(MODEL_EDGES[0] == 0.0 and MODEL_EDGES[-1] == fi.WINDOW_S, "grid spans the window")
    ok(all(b > a for a, b in zip(MODEL_EDGES, MODEL_EDGES[1:])), "edges strictly increase")
    ok(model_bin(0.0) == 0 and model_bin(59.9) == 0, "first body bin is 60 s wide")
    ok(model_bin(60.0) == 1, "body boundary is half-open on the left")
    ok(model_bin(239.9) == 3 and model_bin(240.0) == 4, "body/terminal split at 240 s")
    ok(model_bin(244.9) == 4 and model_bin(245.0) == 5, "terminal bins are 5 s")
    ok(model_bin(300.0) == N_MODEL_BINS - 1, "window end lands in the last bin")
    _expect("outside window", ValueError, lambda: model_bin(300.1))
    _expect("negative elapsed", ValueError, lambda: model_bin(-1.0))

    # CONTROL: the body bin must span exactly one 60 s period, or its whole
    # justification -- absorbing the unidentified component by construction -- is
    # false. This is the claim the grid choice rests on.
    ok(all(abs((MODEL_EDGES[k + 1] - MODEL_EDGES[k]) - 60.0) < 1e-12
           for k in range(4)), "each body bin must be exactly one 60 s period")

    # --- exposure on an arbitrary edge set, including a straddling gap
    e = [0.0, 60.0, 120.0]
    ok(edge_exposure([], e) == [60.0, 60.0], "no gaps leaves full exposure")
    ok(edge_exposure([(50.0, 70.0)], e) == [50.0, 50.0],
       "a straddling gap must debit BOTH bins, not one")
    ok(edge_exposure([(0.0, 200.0)], e) == [0.0, 0.0], "a covering gap zeroes exposure")

    # --- spearman
    ok(abs(spearman_rho([5, 4, 3, 2, 1]) + 1.0) < 1e-9, "monotone decreasing -> -1")
    ok(abs(spearman_rho([1, 2, 3, 4, 5]) - 1.0) < 1e-9, "monotone increasing -> +1")
    ok(abs(spearman_rho([1, 1, 1, 1, 1])) < 1e-9 or
       math.isnan(spearman_rho([1, 1, 1, 1, 1])), "flat -> 0 or nan")

    # --- terminal statistic: synthetic ARTEFACT vs synthetic TWAP.
    # Both directions are required. A rule that only fires one way is a rule that
    # cannot be wrong.
    expo = [FINE_W] * FINE_BINS
    art = []
    for k in range(FINE_BINS):
        art.append(1000.0 * math.exp(-0.30 * (k % FINE_PER_MINUTE) / FINE_PER_MINUTE))
    st_art = terminal_stats(art, expo)
    ok(abs(st_art["amplitude_ratio"] - 1.0) < 0.05,
       f"a periodic artefact must give ratio ~1, got {st_art['amplitude_ratio']:.3f}")

    twap = []
    for k in range(FINE_BINS):
        base = 1000.0 * math.exp(-0.30 * (k % FINE_PER_MINUTE) / FINE_PER_MINUTE)
        if k >= (N_MINUTES - 1) * FINE_PER_MINUTE:
            j = k - (N_MINUTES - 1) * FINE_PER_MINUTE
            base *= math.exp(-2.0 * j / FINE_PER_MINUTE)
        twap.append(base)
    st_twap = terminal_stats(twap, expo)
    ok(st_twap["amplitude_ratio"] > TERM_MATERIAL_BAR,
       f"a terminal-specific collapse must exceed the bar, got "
       f"{st_twap['amplitude_ratio']:.3f}")
    ok(st_twap["terminal_spearman_rho"] <= TERM_MONOTONE_RHO,
       "a smooth terminal decay must read as monotone")

    # --- verdict rule, both directions plus the power floor
    def pc(ratio_lo, ratio_hi, rho, n=1000.0):
        return {"min_terminal_bin_events": n,
                "stats": {"terminal_spearman_rho": rho},
                "bootstrap": {"amplitude_ratio_ci95": [ratio_lo, ratio_hi]}}

    v = terminal_verdict({"btc": pc(4.0, 6.0, -0.95), "eth": pc(3.5, 5.0, -0.90)})
    ok(v["verdict"] == "TWAP_FAVOURED", f"clear terminal-specific case, got {v}")
    ok("does NOT establish" in " ".join(v["notes"]),
       "TWAP_FAVOURED must carry the cannot-establish caveat")

    v = terminal_verdict({"btc": pc(0.8, 1.2, -0.95), "eth": pc(0.9, 1.3, -0.90)})
    ok(v["verdict"] == "ARTEFACT_FAVOURED", f"comparable-amplitude case, got {v}")

    v = terminal_verdict({"btc": pc(4.0, 6.0, -0.95), "eth": pc(0.9, 1.3, -0.90)})
    ok(v["verdict"] == "STILL_CONFOUNDED", "verdict coins disagreeing -> confounded")

    v = terminal_verdict({"btc": pc(4.0, 6.0, -0.20), "eth": pc(3.5, 5.0, -0.10)})
    ok(v["verdict"] == "STILL_CONFOUNDED",
       "a large but NON-MONOTONE terminal effect must not read as TWAP")

    v = terminal_verdict({"btc": pc(4.0, 6.0, -0.95, n=5.0),
                          "eth": pc(3.5, 5.0, -0.90, n=5.0)})
    ok(v["verdict"] == "UNRESOLVED" and not v["supports_twap"],
       "underpowered must NOT return the attractive answer")

    # --- profile recovery on a known non-uniform grid
    cnt = [0.0] * N_MODEL_BINS
    ex = [MODEL_EDGES[k + 1] - MODEL_EDGES[k] for k in range(N_MODEL_BINS)]
    for k in range(N_MODEL_BINS):
        cnt[k] = 2.0 * ex[k]                     # exactly 2 arrivals/second
    prof = fi.profile_ratio([(cnt, ex)])
    ok(all(abs(p - 2.0) < 1e-9 for p in prof),
       "a constant-rate window must give a FLAT profile on the non-uniform grid")

    print(f"flow_grid_terminal selftest: {checks} checks OK")
    return 0


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("grid", "terminal", "both"),
                    default="both")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    if args.cmd in ("grid", "both"):
        g = grid(n_boot=args.n_boot)
        OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
        OUT_GRID.write_text(json.dumps(g, indent=1))
        print(f"[grid] non-uniform f_r, {N_MODEL_BINS} bins "
              f"({int(BODY_END_S/BODY_W)}x{BODY_W:.0f}s body + "
              f"{int((fi.WINDOW_S-BODY_END_S)/TERMINAL_W)}x{TERMINAL_W:.0f}s terminal)")
        print(f"{'coin':6}{'windows':>8}{'trades':>9}{'micro%':>8}{'primary':>14}"
              f"{'shape(cnt)':>12}{'shape(notl)':>12}{'term x(cnt)':>12}{'term x(notl)':>13}")
        for c, d in g["coins"].items():
            print(f"{c:6}{d['n_windows']:>8}{d['n_trades']:>9}"
                  f"{d['micro_event_share']*100:>7.1f}%{d['primary_weighting']:>14}"
                  f"{d['count']['shape_ratio']:>12.2f}{d['notional']['shape_ratio']:>12.2f}"
                  f"{(d['count']['terminal_collapse_ratio'] or 0):>12.2f}"
                  f"{(d['notional']['terminal_collapse_ratio'] or 0):>13.2f}")
        print(f"[grid] wrote {OUT_GRID}")

    if args.cmd in ("terminal", "both"):
        t = terminal(n_boot=args.n_boot)
        OUT_TERM.parent.mkdir(parents=True, exist_ok=True)
        OUT_TERM.write_text(json.dumps(t, indent=1))
        print(f"\n[terminal] {t['test']}")
        print(f"{'coin':6}{'body range':>12}{'term range':>12}{'ratio':>8}"
              f"{'ratio CI95':>20}{'rho':>8}{'min bin':>9}")
        for c, d in t["coins"].items():
            s, b = d["stats"], d["bootstrap"]
            lo, hi = b["amplitude_ratio_ci95"]
            ci = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else "[--, --]"
            print(f"{c:6}{s['body_mean_log_range']:>12.3f}{s['terminal_log_range']:>12.3f}"
                  f"{s['amplitude_ratio']:>8.2f}{ci:>20}"
                  f"{s['terminal_spearman_rho']:>8.2f}{d['min_terminal_bin_events']:>9.0f}")
        v = t["verdict"]
        print(f"\n[terminal] VERDICT: {v['verdict']} — {v['reason']}")
        for n in v["notes"]:
            print(f"           {n}")
        print(f"[terminal] wrote {OUT_TERM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
