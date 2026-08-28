#!/usr/bin/env python3
"""Iteration 011 — conditional signed-value decomposition. DEVELOPMENT SLICE 1.

Preregistration: plans/ITER011_CONDITIONAL_VALUE_PREREGISTRATION.md, FROZEN by
user ruling R-232 (3b71d3e). This module implements the target builder and the
four heads. It fits nothing here and scores nothing here.

THE ESTIMANDS ARE THE FROZEN ONES, reproduced from parent plan §2.1:

    p_harm(x) = P(V_cancel > 0 | preventable fill, x)
    m_harm(x) = E[V_cancel | V_cancel > 0, preventable fill, x]
    m_good(x) = E[-V_cancel | V_cancel < 0, preventable fill, x]
    conditional_cancel_value(x)
        = p_harm(x) * m_harm(x) - (1 - p_harm(x)) * m_good(x)
    expected_cancel_value(x) = p_fill(x) * conditional_cancel_value(x)

Q4 IS COMPOSED, NEVER FITTED (prereg §6). Composing is the hypothesis under
test; fitting Q4 directly would answer a different question and silently rescue
a failed decomposition. There is deliberately no fit_q4 in this module.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

import phase2_declaration as D

# Features whose CONSTRUCTION reads the fill outcome. Inadmissible regardless of
# IC (prereg §7): the fair-price module owns E[Y|state]; a toxicity feature that
# absorbs an E[Y|state, FILLED] price puts adverse selection in both terms.
FILL_CONDITIONED_TOKENS = (
    "filled", "fill_", "_fill", "postfill", "post_fill", "realized_fill",
    "markout", "pnl", "outcome", "future", "ahead_value",
)
# THE NAME SCREEN IS A PROXY, NOT THE INVARIANT. The real rule is: a feature's
# CONSTRUCTION must not read anything at or after the decision time. A name
# cannot decide that -- the same lesson as path-prefix vs runtime bytes in
# R-230(3). So the screen is deliberately over-broad and every hit must be
# either REFUSED or admitted through the reviewed register below, which carries
# the construction evidence that justified it.
#
# Adding to this register is a deliberate act with a stated reason. It is NOT an
# allowlist tuned until the suite passes: an entry without construction evidence
# is a fence with a hole in it.
FENCE_REVIEWED = {
    "any_fill_ahead": "the valuation GATE, not a feature; never in the pin",
    "queue_ahead": "queue position at decision time; 'ahead' is spatial",
    "qahead": "as above",
}
# DA's state schema names market trade flow with 'fill'. Reviewed against the
# construction in harmful_state_features (`lo = t - ms/1000.0`, walking
# BACKWARDS from `last_at(trades_t, t)`): these aggregate trades in [t-ms, t],
# strictly BEFORE the decision instant, and describe the TAKER FLOW THAT HITS
# THIS MAKER -- not whether this order was filled. Decision-time, admissible.
#
# NAMING HAZARD, worth stating: `same_side_fill_share` reads naturally as "share
# of MY order filled". It is not. A future feature that genuinely reads this
# order's fill outcome could be named alike and slip past a reader; the screen
# will still flag it, and it must be refused rather than added here.
for _ms in (50, 250, 1000):
    for _t in ("same_side_fill_share", "same_side_fill_size",
               "opp_side_fill_size", "fill_share_missing"):
        FENCE_REVIEWED[f"{_t}_{_ms}ms"] = (
            "market taker-flow aggregate over [t-%dms, t]; backward-looking by "
            "construction (harmful_state_features), NOT this order's fill "
            "outcome" % _ms)


def signed_v_cancel(row: dict, latency_ms: int = None) -> float:
    """V_cancel: signed value of cancelling, from the neutral-path row.

    POSITIVE = cancelling would have avoided harm.
    NEGATIVE = cancelling would have forfeited a good fill.

    Read from the exposure row's own latency bucket. `harmful_exposure_rows`
    owns BOTH the latency cut (only tranches at t >= t_start + L/1000 are
    valued) and the valuation gate (any_fill_ahead). This function must not
    re-derive either -- two definitions of a valuation gate is one too many
    (R-228(12))."""
    L = str(D.TARGET_LATENCY_MS if latency_ms is None else latency_ms)
    lat = row.get("latency") or {}
    if not row.get("any_fill_ahead") or L not in lat:
        return 0.0
    return float(lat[L].get("preventable_value_cents", 0.0))


def preventable(row: dict, latency_ms: int = None) -> bool:
    """A PREVENTABLE fill in the latency sense, not the colloquial one."""
    L = str(D.TARGET_LATENCY_MS if latency_ms is None else latency_ms)
    lat = row.get("latency") or {}
    return bool(row.get("any_fill_ahead")) and \
        float(lat.get(L, {}).get("preventable_shares", 0.0)) > 0


def head_populations(rows, latency_ms: int = None) -> dict:
    """The four heads' populations, each named, with its exclusions counted.

    Rule 4: exclusions are STATUSES, never silent drops."""
    L = latency_ms
    q1 = list(rows)                                   # fill arrival: all rows
    prev = [r for r in rows if preventable(r, L)]     # the conditional base
    v = {id(r): signed_v_cancel(r, L) for r in prev}
    harm = [r for r in prev if v[id(r)] > 0]
    good = [r for r in prev if v[id(r)] < 0]
    zero = [r for r in prev if v[id(r)] == 0.0]
    return {
        "q1_arrival": q1,
        "q2_sign_base": prev,
        "q3_harm": harm,
        "q3_good": good,
        "zero_value_preventable": zero,
        "counts": {"all_rows": len(q1), "preventable": len(prev),
                   "v_positive": len(harm), "v_negative": len(good),
                   "v_zero": len(zero)},
    }


def zero_mass_diagnostic(pops: dict) -> dict:
    """THE FROZEN FORMULA IS EXACT ONLY IF P(V_cancel == 0 | preventable) == 0.

    E[V] = P(V>0)E[V|V>0] + P(V<0)E[V|V<0], so the identity

        p_harm*m_harm - (1 - p_harm)*m_good

    uses (1 - p_harm) where the algebra wants P(V<0). Those differ by exactly
    P(V == 0). If zero-value preventable fills exist, the composed value
    OVERSTATES the good-side term by m_good * P(V==0) and is biased DOWNWARD.

    This is computed and reported rather than silently corrected: the
    preregistration is FROZEN and reproduces §2.1 verbatim, so BE does not get
    to quietly substitute a different estimator. The number below is what the
    user needs in order to decide whether the frozen form needs an amendment."""
    n = pops["counts"]["preventable"]
    z = pops["counts"]["v_zero"]
    frac = (z / n) if n else 0.0
    return {
        "p_zero_given_preventable": frac,
        "n_zero": z, "n_preventable": n,
        "frozen_formula_is_exact": z == 0,
        "bias_direction_if_nonzero": "composed conditional value is biased "
                                     "DOWNWARD by m_good * P(V==0), because "
                                     "(1 - p_harm) counts zero-value rows into "
                                     "the good-side term while m_good is "
                                     "estimated only on V < 0",
        "action": "REPORT, do not silently correct. The frozen preregistration "
                  "reproduces parent §2.1 verbatim; amending an estimand after "
                  "seeing data is the thing the freeze exists to prevent. If "
                  "this fraction is material the USER amends the prereg.",
    }


def empirical_heads(pops: dict, latency_ms: int = None) -> dict:
    """Unconditional empirical values of the four estimands. NOT a model.

    These are the population quantities the fitted heads must reproduce; a head
    that cannot match its own population mean on the training data is broken
    before any comparison."""
    L = latency_ms
    prev = pops["q2_sign_base"]
    harm = [signed_v_cancel(r, L) for r in pops["q3_harm"]]
    good = [-signed_v_cancel(r, L) for r in pops["q3_good"]]
    p_harm = (len(harm) / len(prev)) if prev else 0.0
    m_harm = (math.fsum(harm) / len(harm)) if harm else 0.0
    m_good = (math.fsum(good) / len(good)) if good else 0.0
    return {"p_harm": p_harm, "m_harm": m_harm, "m_good": m_good,
            "n_harm": len(harm), "n_good": len(good),
            "conditional_cancel_value":
                p_harm * m_harm - (1.0 - p_harm) * m_good}


def compose_expected_cancel_value(p_fill, p_harm, m_harm, m_good) -> float:
    """Q4, COMPOSED from the heads. Never fitted (prereg §6)."""
    return float(p_fill) * (float(p_harm) * float(m_harm)
                            - (1.0 - float(p_harm)) * float(m_good))


def assert_no_fill_conditioned_features(features) -> dict:
    """The E[Y|state] fence, asserted BY NAME against the pinned schema.

    Prereg §7 requires this be a PREDICATE, not a promise. A feature whose
    construction reads the fill outcome is inadmissible regardless of its IC."""
    bad = {}
    for f in features:
        if f in FENCE_REVIEWED:
            continue
        hits = [t for t in FILL_CONDITIONED_TOKENS if t in f.lower()]
        if hits:
            bad[f] = hits
    if bad:
        raise RuntimeError(
            f"REFUSED: fill-conditioned feature(s) in the iteration-011 set: "
            f"{bad}. The fair-price module owns E[Y|state]; a toxicity feature "
            f"that absorbs an E[Y|state, FILLED] price puts adverse selection "
            f"in BOTH terms and counts it twice (parent §2.2). Inadmissible "
            f"regardless of IC. If one of these is decision-time by "
            f"construction, it is REVIEWED into FENCE_REVIEWED with the "
            f"evidence -- never allowlisted to make a suite pass.")
    return {"checked": len(list(features)), "fence": "clean",
            "reviewed_admissions": len(FENCE_REVIEWED)}


UNDERPOWERED_MIN_N = 100          # per prereg §3: reported, never omitted


def head_power(pops: dict) -> dict:
    """A head with too few conditional observations is reported UNDERPOWERED,
    never omitted (prereg §3). Q3 is the head most likely to be silently
    skipped, so its n is surfaced explicitly."""
    c = pops["counts"]
    return {
        "q1_arrival": {"n": c["all_rows"],
                       "underpowered": c["all_rows"] < UNDERPOWERED_MIN_N},
        "q2_sign": {"n": c["preventable"],
                    "underpowered": c["preventable"] < UNDERPOWERED_MIN_N},
        "q3_m_harm": {"n": c["v_positive"],
                      "underpowered": c["v_positive"] < UNDERPOWERED_MIN_N},
        "q3_m_good": {"n": c["v_negative"],
                      "underpowered": c["v_negative"] < UNDERPOWERED_MIN_N},
        "min_n": UNDERPOWERED_MIN_N,
    }


def selftest() -> int:
    fails = []

    def ok(c, label):
        print(f"  {'PASS' if c else 'FAIL'}  {label}")
        if not c:
            fails.append(label)

    L = str(D.TARGET_LATENCY_MS)

    def row(v, shares=1.0, fill=True):
        return {"any_fill_ahead": fill,
                "latency": {L: {"preventable_value_cents": v,
                                "preventable_shares": shares,
                                "stale_shares": 0.0}}}

    # --- target: sign convention and the gate ---
    ok(signed_v_cancel(row(5.0)) == 5.0, "V_cancel positive = harm avoided")
    ok(signed_v_cancel(row(-3.0)) == -3.0, "V_cancel negative = good fill lost")
    ok(signed_v_cancel(row(5.0, fill=False)) == 0.0,
       "no fill ahead -> 0; the GATE is the exposure row's, not re-derived here")
    ok(signed_v_cancel({}) == 0.0, "a row with no latency block is 0, not a crash")
    ok(not preventable(row(5.0, shares=0.0)),
       "zero preventable shares is NOT a preventable fill")

    # --- populations partition, no silent drops ---
    rows = [row(5.0), row(-3.0), row(0.0), row(2.0, fill=False)]
    pops = head_populations(rows)
    c = pops["counts"]
    ok(c["preventable"] == 3, "the no-fill row is excluded from the base")
    ok(c["v_positive"] + c["v_negative"] + c["v_zero"] == c["preventable"],
       "harm/good/zero PARTITION the preventable base (rule 4: no silent drops)")

    # --- THE ZERO-MASS FINDING ---
    z = zero_mass_diagnostic(pops)
    ok(z["n_zero"] == 1 and not z["frozen_formula_is_exact"],
       "zero-value preventable fills are DETECTED and reported, not absorbed")
    zc = zero_mass_diagnostic(head_populations([row(5.0), row(-3.0)]))
    ok(zc["frozen_formula_is_exact"],
       "with no zero-value rows the frozen formula is exact and says so")

    # --- empirical heads ---
    e = empirical_heads(head_populations([row(4.0), row(6.0), row(-2.0)]))
    ok(abs(e["p_harm"] - 2 / 3) < 1e-12, "p_harm = P(V>0 | preventable)")
    ok(abs(e["m_harm"] - 5.0) < 1e-12, "m_harm = E[V | V>0]")
    ok(abs(e["m_good"] - 2.0) < 1e-12, "m_good = E[-V | V<0], POSITIVE by defn")
    ok(abs(e["conditional_cancel_value"] - (2/3*5.0 - 1/3*2.0)) < 1e-12,
       "conditional value composes per the FROZEN formula")

    # --- Q4 is composed, and cannot be fitted from this module ---
    ok(compose_expected_cancel_value(0.5, 0.6, 10.0, 4.0)
       == 0.5 * (0.6 * 10.0 - 0.4 * 4.0), "Q4 composes from the four heads")
    ok(not any(n.startswith("fit_q4") or n == "fit_expected_cancel_value"
               for n in globals()),
       "there is NO direct Q4 fitter: composing IS the hypothesis under test")

    # --- the fence: a known-bad it must refuse, and a clean set it must pass ---
    try:
        assert_no_fill_conditioned_features(["spread", "filled_notional_5s"])
        ok(False, "the fence REFUSES a fill-conditioned feature")
    except RuntimeError as ex:
        ok("filled_notional_5s" in str(ex),
           "the fence REFUSES a fill-conditioned feature, naming it")
    try:
        assert_no_fill_conditioned_features(["spread", "markout_1s"])
        ok(False, "the fence REFUSES an outcome feature (markout)")
    except RuntimeError:
        ok(True, "the fence REFUSES an outcome feature (markout)")
    ok(assert_no_fill_conditioned_features(
        ["spread", "imbalance", "queue_ahead", "any_fill_ahead"])["fence"]
       == "clean",
       "the fence PASSES a clean set and admits the gate by review "
       "(a fence that refuses everything is not a fence)")

    for _bad in ("my_fill_share_realized", "order_filled_flag",
                 "post_fill_markout", "same_side_fill_share_9999ms"):
        try:
            assert_no_fill_conditioned_features([_bad])
            ok(False, f"the fence still REFUSES {_bad}")
        except RuntimeError:
            ok(True, f"the fence still REFUSES {_bad} (review is per-NAME, "
                     f"so a lookalike does not inherit an admission)")

    # --- the fence runs against the REAL pinned schema ---
    try:
        import phase2_state_schema_freeze as PIN
        feats = PIN.build_pin()["features_in_order"]
        assert_no_fill_conditioned_features(feats)
        ok(True, f"the PINNED schema ({len(feats)} features) passes the fence")
    except RuntimeError as ex:
        ok(False, f"the PINNED schema FAILS the fence: {str(ex)[:120]}")
    except ImportError:
        ok(False, "could not load the pinned schema to check it")

    # --- underpowered is REPORTED, not omitted ---
    hp = head_power(head_populations([row(5.0)]))
    ok(hp["q3_m_good"]["underpowered"] and hp["q3_m_good"]["n"] == 0,
       "an EMPTY magnitude head is reported UNDERPOWERED with its n, "
       "not omitted (prereg §3)")

    print(f"\n{'ITER011 SLICE-1 SELFTEST GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(selftest())
