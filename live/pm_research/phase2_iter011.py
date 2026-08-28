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
    # A1.2 (FROZEN): `any_fill_ahead` was HERE, admitted as "the valuation GATE,
    # not a feature; never in the pin". Both clauses are true and neither is a
    # defence — it is an OUTCOME field, and the fence exists to ban outcome
    # fields BY NAME rather than trust they never reach the pin. A reviewed
    # admission is a standing permission, and that one permitted exactly the
    # class the fence was built to stop. Its use as the valuation GATE is
    # unaffected: the gate is not a feature and never passes through here.
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


class MalformedRow(RuntimeError):
    """A row whose valuation inputs cannot be trusted. A1.3 (FROZEN).

    Carries a NAMED status so the refusal can be counted rather than absorbed:
    a zero that means 'absent' and a zero that means 'no harm' must not be the
    same number."""

    def __init__(self, status: str, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"[{status}] {detail}")


def _num(x):
    """A finite real, or None. bool is rejected: True would arithmetic as 1.0."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(float(x)) else None


def validate_row(row: dict, latency_ms: int = None) -> dict:
    """Strict validation of one row's valuation inputs. A1.3 (FROZEN).

    Fail-open was the defect: a missing or inconsistent field became a clean 0.0
    or a 'no fill' row indistinguishable from a genuine one. Every failure below
    is a NAMED status the caller must count."""
    L = str(D.TARGET_LATENCY_MS if latency_ms is None else latency_ms)
    if not isinstance(row, dict):
        raise MalformedRow("NOT_A_ROW", f"got {type(row).__name__}")
    if "any_fill_ahead" not in row:
        raise MalformedRow("MISSING_GATE", "no any_fill_ahead on the row")
    gate = row["any_fill_ahead"]
    if not isinstance(gate, bool):
        raise MalformedRow("NON_BOOLEAN_GATE", f"any_fill_ahead={gate!r}")
    lat = row.get("latency")
    if lat is None:
        # a row with NO fill ahead legitimately carries no latency block
        if gate:
            raise MalformedRow("MISSING_LATENCY",
                               "gate says a fill is ahead but there is no "
                               "latency block to value it from")
        return {"ok": True, "gate": False, "value": 0.0, "shares": 0.0}
    if not isinstance(lat, dict):
        raise MalformedRow("MALFORMED_LATENCY", f"latency={type(lat).__name__}")
    if L not in lat:
        raise MalformedRow("MISSING_LATENCY_BUCKET",
                           f"no bucket {L!r}; buckets present: {sorted(lat)}")
    b = lat[L]
    if not isinstance(b, dict):
        raise MalformedRow("MALFORMED_BUCKET", f"bucket {L} is {type(b).__name__}")
    val = _num(b.get("preventable_value_cents"))
    sh = _num(b.get("preventable_shares"))
    if val is None:
        raise MalformedRow("NON_NUMERIC_VALUE",
                           f"preventable_value_cents="
                           f"{b.get('preventable_value_cents')!r}")
    if sh is None:
        raise MalformedRow("NON_NUMERIC_SHARES",
                           f"preventable_shares={b.get('preventable_shares')!r}")
    if sh < 0:
        raise MalformedRow("NEGATIVE_SHARES", f"preventable_shares={sh}")
    if val != 0.0 and sh == 0.0:
        raise MalformedRow(
            "VALUE_WITHOUT_SHARES",
            f"preventable_value_cents={val} with preventable_shares=0. A "
            f"non-zero preventable value requires preventable shares; the pair "
            f"is inconsistent and was previously accepted as a NO-FILL row.")
    return {"ok": True, "gate": gate, "value": val, "shares": sh}


def signed_v_cancel(row: dict, latency_ms: int = None) -> float:
    """V_cancel: signed value of cancelling, from the neutral-path row.

    POSITIVE = cancelling would have avoided harm.
    NEGATIVE = cancelling would have forfeited a good fill.

    Read from the exposure row's own latency bucket. `harmful_exposure_rows`
    owns BOTH the latency cut (only tranches at t >= t_start + L/1000 are
    valued) and the valuation gate (any_fill_ahead). This function must not
    re-derive either -- two definitions of a valuation gate is one too many
    (R-228(12))."""
    v = validate_row(row, latency_ms)
    return v["value"] if v["gate"] else 0.0


def preventable(row: dict, latency_ms: int = None) -> bool:
    """A PREVENTABLE fill in the latency sense, not the colloquial one."""
    v = validate_row(row, latency_ms)
    return bool(v["gate"]) and v["shares"] > 0


def head_populations(rows, latency_ms: int = None) -> dict:
    """The four heads' populations, each named, with its exclusions counted.

    Rule 4: exclusions are STATUSES, never silent drops."""
    L = latency_ms
    # A1.3 (FROZEN): materialise ONCE. `rows` may be a generator, and the
    # previous form consumed it with list(rows) and then iterated it AGAIN —
    # yielding preventable=0 from a non-empty population, silently.
    q1 = list(rows)
    prev, refused = [], {}
    vals = {}
    for i, r in enumerate(q1):
        try:
            if preventable(r, L):
                prev.append(i)
                vals[i] = signed_v_cancel(r, L)
        except MalformedRow as e:
            refused[e.status] = refused.get(e.status, 0) + 1
    harm = [q1[i] for i in prev if vals[i] > 0]
    good = [q1[i] for i in prev if vals[i] < 0]
    zero = [q1[i] for i in prev if vals[i] == 0.0]
    prev = [q1[i] for i in prev]
    return {
        "q1_arrival": q1,
        "q2_sign_base": prev,
        "q3_harm": harm,
        "q3_good": good,
        "zero_value_preventable": zero,
        "counts": {"all_rows": len(q1), "preventable": len(prev),
                   "v_positive": len(harm), "v_negative": len(good),
                   "v_zero": len(zero),
                   "refused_malformed": sum(refused.values())},
        # rule 4: exclusions are STATUSES, never silent drops
        "refused_by_status": dict(sorted(refused.items())),
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

    def row(v, shares=1.0, fill=True, v_override=None):
        """Build a CONSISTENT row. Under A1.3 a value without shares is a
        refusal, so the helper must not manufacture that pair by accident."""
        val = v if v_override is None else v_override
        return {"any_fill_ahead": fill,
                "latency": {L: {"preventable_value_cents": val,
                                "preventable_shares": shares,
                                "stale_shares": 0.0}}}

    # --- target: sign convention and the gate ---
    ok(signed_v_cancel(row(5.0)) == 5.0, "V_cancel positive = harm avoided")
    ok(signed_v_cancel(row(-3.0)) == -3.0, "V_cancel negative = good fill lost")
    ok(signed_v_cancel({"any_fill_ahead": False}) == 0.0,
       "no fill ahead -> 0; the GATE is the exposure row's, not re-derived here")

    # A1.3 (FROZEN): these previously FAILED OPEN. The old falsifier here
    # asserted `signed_v_cancel({}) == 0.0` — "0, not a crash" — which encoded
    # the fail-open behaviour AS CORRECT. A test can enshrine a defect as the
    # specification, and this one did.
    for _lbl, _r, _st in (
            ("no gate field", {}, "MISSING_GATE"),
            ("gate true, no latency", {"any_fill_ahead": True}, "MISSING_LATENCY"),
            ("non-numeric value", {"any_fill_ahead": True, "latency": {
                L: {"preventable_value_cents": None,
                    "preventable_shares": 1.0}}}, "NON_NUMERIC_VALUE"),
            ("value>0 with shares==0", {"any_fill_ahead": True, "latency": {
                L: {"preventable_value_cents": 5.0,
                    "preventable_shares": 0.0}}}, "VALUE_WITHOUT_SHARES"),
            ("negative shares", {"any_fill_ahead": True, "latency": {
                L: {"preventable_value_cents": 1.0,
                    "preventable_shares": -1.0}}}, "NEGATIVE_SHARES"),
            ("non-boolean gate", {"any_fill_ahead": 1, "latency": {
                L: {"preventable_value_cents": 1.0,
                    "preventable_shares": 1.0}}}, "NON_BOOLEAN_GATE")):
        try:
            signed_v_cancel(_r)
            ok(False, f"A1.3 REFUSES a malformed row: {_lbl}")
        except MalformedRow as _e:
            ok(_e.status == _st,
               f"A1.3 REFUSES a malformed row: {_lbl} -> {_e.status}")
    ok(signed_v_cancel(row(5.0)) == 5.0,
       "A1.3 ACCEPTS a well-formed row (strictness is not a wall)")
    ok(not preventable(row(5.0, shares=0.0, v_override=0.0)),
       "zero preventable shares with zero value is NOT a preventable fill")

    # --- populations partition, no silent drops ---
    rows = [row(5.0), row(-3.0), row(0.0), row(2.0, fill=False)]
    pops = head_populations(rows)
    c = pops["counts"]
    ok(c["preventable"] == 3, "the no-fill row is excluded from the base")
    _g = (row(5.0) for _ in range(3))
    ok(head_populations(_g)["counts"]["preventable"] == 3,
       "A1.3 head_populations builds ONCE: a GENERATOR input is not "
       "double-consumed (it previously yielded preventable=0 from 3 rows)")
    _mixed = head_populations([row(5.0), {"any_fill_ahead": True}, row(-2.0)])
    ok(_mixed["counts"]["refused_malformed"] == 1
       and _mixed["refused_by_status"] == {"MISSING_LATENCY": 1},
       "A1.3 malformed rows are COUNTED BY STATUS, never silently dropped "
       "(rule 4)")
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
    try:
        assert_no_fill_conditioned_features(["spread", "any_fill_ahead"])
        ok(False, "A1.2 the fence NAME-BANS any_fill_ahead")
    except RuntimeError:
        ok(True, "A1.2 the fence NAME-BANS any_fill_ahead — it is an OUTCOME "
                 "field, and 'it is the gate, not a feature' was true but not "
                 "a defence")
    ok(assert_no_fill_conditioned_features(
        ["spread", "imbalance", "queue_ahead"])["fence"]
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

    # ---------------------------------------------------------- SLICE 2 ---
    # A metric that cannot distinguish a perfect ranker from an inverted one is
    # not a metric. Each is driven at both extremes and at chance.
    ok(auc([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) == 1.0, "AUC: perfect = 1.0")
    ok(auc([0.4, 0.3, 0.2, 0.1], [0, 0, 1, 1]) == 0.0,
       "AUC: INVERTED = 0.0 (a metric blind to sign is not a metric)")
    ok(auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5,
       "AUC: all-ties = 0.5 exactly, not a divide-by-zero")
    ok(auc([0.1, 0.2], [1, 1]) is None,
       "AUC: one-class returns None rather than a number nobody can read")
    ok(brier([1.0, 0.0], [1, 0]) == 0.0, "Brier: perfect = 0")
    ok(abs(brier([0.5, 0.5], [1, 0]) - 0.25) < 1e-12, "Brier: uninformative = 0.25")
    ok(abs(mae([1.0, 2.0], [2.0, 4.0]) - 1.5) < 1e-12, "MAE")
    ok(abs(calibration_slope([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-12,
       "calibration slope: y=x is 1.0")
    ok(abs(calibration_slope([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) - 2.0) < 1e-12,
       "calibration slope: y=2x is 2.0")
    ok(calibration_slope([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None,
       "calibration slope: a CONSTANT predictor has no slope -> None, not 0.0 "
       "('no information to calibrate' is a different finding from 'badly "
       "calibrated')")

    # head_report: n travels with every number; underpowered is a STATUS
    r = head_report("Q3_m_harm", "magnitude", [1.0] * 3, [1.0] * 3)
    ok(r["status"] == UNDERPOWERED and r["n"] == 3,
       "a thin magnitude head is reported UNDERPOWERED with its n, not dropped")
    r2 = head_report("Q2_sign", "probability",
                     [i / 200 for i in range(200)], [i % 2 for i in range(200)])
    ok(r2["status"] == "OK" and r2["auc"] is not None,
       "a powered probability head reports OK with its AUC")
    try:
        head_report("x", "not_a_kind", [1.0], [1.0]); ok(False, "unknown kind refuses")
    except RuntimeError:
        ok(True, "an unknown head kind REFUSES rather than silently reporting")

    # the four-head report must carry FAILURES, not just winners
    fr = four_head_report(
        head_report("Q1_arrival", "probability", [0.1] * 300, [0] * 150 + [1] * 150),
        head_report("Q2_sign", "probability", [0.1] * 5, [0, 1, 0, 1, 0]),
        head_report("Q3_m_harm", "magnitude", [1.0] * 4, [1.0] * 4),
        head_report("Q3_m_good", "magnitude", [1.0] * 400, [1.0] * 400))
    ok(fr["all_heads_reported"], "ALL FOUR heads appear in the report")
    ok(fr["underpowered_heads"] == ["Q2_sign", "Q3_m_harm"],
       "the report NAMES which heads are underpowered rather than omitting them")
    ok("may NOT advance" in fr["advancement_rule"],
       "the report carries R-232 9.2: Q4 alone cannot advance a candidate")

    # ---------------------------------------------------------- SLICE 3 ---
    # THE DETERMINISM PROPERTY, asserted as the INVERSE of the control that
    # exposed R-234: two different insertion orders must now give IDENTICAL p.
    base = {f"w{i}": v for i, v in enumerate(
        [90.0, 70.0, 55.0, 40.0, 30.0, 25.0, 20.0, 15.0, 10.0, 8.0,
         -5.0, -7.0, -9.0, -12.0, -14.0, -18.0, -22.0, -28.0, -35.0, -44.0])}
    import random as _r
    fwd = sign_flip_null(base, n_perm=400, seed=7)
    rev = sign_flip_null(dict(reversed(list(base.items()))), n_perm=400, seed=7)
    shuf = []
    for _s in (1, 2, 3):
        k = list(base); _r.Random(_s).shuffle(k)
        shuf.append(sign_flip_null({x: base[x] for x in k},
                                   n_perm=400, seed=7)["p_two_sided"])
    ok(fwd["p_two_sided"] == rev["p_two_sided"] == shuf[0] == shuf[1] == shuf[2],
       "sign-flip p is IDENTICAL across insertion orders (R-234 fixed AT "
       "CONSTRUCTION: the seed pins the stream, sorting pins what it is "
       "applied to)")
    ok(fwd["observed"] == rev["observed"],
       "the observed statistic was never order-dependent; only the NULL was")
    ok(fwd["p_two_sided"] > 0, "permutation p is never exactly 0 ((ge+1)/(n+1))")

    # the null must FIRE on a real effect and NOT on a balanced one
    strong = sign_flip_null({f"w{i}": 100.0 for i in range(40)}, n_perm=400, seed=7)
    bal = sign_flip_null({f"w{i}": (100.0 if i % 2 else -100.0)
                          for i in range(40)}, n_perm=400, seed=7)
    ok(strong["p_two_sided"] < 0.01, "the null FLAGS a 40-unit one-sided effect")
    ok(bal["p_two_sided"] > 0.2, "the null does NOT flag a balanced sample")

    # Holm across the declared family
    fam = declared_family()
    ok(fam["n_cells"] == 24 and len(set(fam["cells"])) == 24,
       "the declared family is 2 arms x 4 heads x 3 budgets = 24 DISTINCT cells "
       "(R-232 9.1)")
    h = holm({"a": 0.001, "b": 0.02, "c": 0.5})
    ok(abs(h["a"] - 0.003) < 1e-12 and abs(h["b"] - 0.04) < 1e-12,
       "Holm scales by (m - rank), not uniformly by m")
    hm = holm({"a": 0.02, "b": 0.021, "c": 0.9})
    ok(hm["a"] <= hm["b"] <= hm["c"],
       "Holm is MONOTONE step-down (an unenforced version can invert)")
    ok(all(v <= 1.0 for v in holm({"a": 0.9, "b": 0.95}).values()),
       "Holm never exceeds 1.0")

    # cluster disclosure must say WEAKER when it is weaker
    cd = cluster_disclosure(0, "window")
    ok(cd["weaker_than_ruled"] and not cd["intervals_claimable"]
       and "OPTIMISTIC" in cd["why"],
       "at G=0 the disclosure says the unit is WEAKER than ruled and the "
       "p-values are OPTIMISTIC")
    ok(cluster_disclosure(6, "UTC day")["intervals_claimable"],
       "at G=6 on the ruled unit, intervals become claimable")

    print(f"\n{'ITER011 SELFTEST GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing")
    return 1 if fails else 0




# ---------------------------------------------------------------- SLICE 2 ---
# Head METRICS. Each head is scored on ITS OWN population (prereg §3): Q1 on all
# rows, Q2 on the preventable base, Q3 on the sign-conditional subsets. A metric
# computed on the wrong population is the quietest way to make a failed head
# look like a working one, so the population is passed in explicitly and the n
# travels with every number.

UNDERPOWERED = "UNDERPOWERED"


def auc(scores, labels) -> float:
    """Rank-based AUC, O(n log n), ties averaged. Returns None if one-class."""
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    npos = sum(1 for _, y in pairs if y)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return None
    ranks, i = [0.0] * n, 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    s = math.fsum(r for r, (_, y) in zip(ranks, pairs) if y)
    return (s - npos * (npos + 1) / 2.0) / (npos * nneg)


def brier(probs, labels) -> float:
    if not probs:
        return None
    return math.fsum((p - (1.0 if y else 0.0)) ** 2 for p, y in
                     zip(probs, labels)) / len(probs)


def mae(pred, actual) -> float:
    if not pred:
        return None
    return math.fsum(abs(p - a) for p, a in zip(pred, actual)) / len(pred)


def calibration_slope(pred, actual) -> float:
    """OLS slope of actual on pred. 1.0 = calibrated.

    None when pred has no spread: a constant predictor has NO slope, and
    returning 0.0 there would report 'badly calibrated' for what is actually
    'no information to calibrate'. Those are different findings."""
    n = len(pred)
    if n < 2:
        return None
    mp = math.fsum(pred) / n
    ma = math.fsum(actual) / n
    sxx = math.fsum((p - mp) ** 2 for p in pred)
    if sxx <= 0.0:
        return None
    sxy = math.fsum((p - mp) * (a - ma) for p, a in zip(pred, actual))
    return sxy / sxx


def head_report(name, kind, pred, actual, min_n: int = UNDERPOWERED_MIN_N) -> dict:
    """One head's metrics WITH its n and power status. Never omitted.

    prereg §3: a head with too few conditional observations is reported
    UNDERPOWERED, never dropped. Q3 is the head most likely to be silently
    skipped, which is exactly why its absence must be representable."""
    n = len(pred)
    out = {"head": name, "kind": kind, "n": n, "min_n": min_n,
           "status": UNDERPOWERED if n < min_n else "OK"}
    if kind == "probability":
        out["auc"] = auc(pred, actual)
        out["brier"] = brier(pred, actual)
        out["n_positive"] = sum(1 for y in actual if y)
        out["one_class"] = out["auc"] is None
    elif kind == "magnitude":
        out["mae"] = mae(pred, actual)
        out["calibration_slope"] = calibration_slope(pred, actual)
        out["no_predictor_spread"] = out["calibration_slope"] is None
    else:
        raise RuntimeError(f"unknown head kind {kind!r}")
    return out


def four_head_report(q1, q2, q3h, q3g) -> dict:
    """All four heads, ALWAYS, including failures (prereg §3).

    A candidate advancing on Q4 while failing Q2 must say so; that combination
    is interesting, not disqualifying, but it must never be PRESENTED as
    toxicity discrimination. Under R-232 9.2 it also cannot advance without
    explicit user sign-off."""
    heads = {"Q1_arrival": q1, "Q2_sign": q2,
             "Q3_m_harm": q3h, "Q3_m_good": q3g}
    underpowered = sorted(k for k, v in heads.items()
                          if v.get("status") == UNDERPOWERED)
    return {
        "heads": heads,
        "all_heads_reported": sorted(heads) == sorted(
            ["Q1_arrival", "Q2_sign", "Q3_m_harm", "Q3_m_good"]),
        "underpowered_heads": underpowered,
        "any_underpowered": bool(underpowered),
        "reporting_rule": "all four are reported whether or not they pass; a "
                          "strong hazard head does not establish toxicity "
                          "discrimination (parent §2.1)",
        "advancement_rule": "Q4 alone may NOT advance a candidate; explicit "
                            "user sign-off required at that time (R-232 9.2)",
    }


# ---------------------------------------------------------------- SLICE 3 ---
# Evaluation harness: the four questions against their DECLARED nulls
# (prereg §5). Nothing here fits or scores; it assembles and adjudicates.

ARMS_011 = ("composed_linear", "composed_lgbm")      # R-232 9.1: TWO arms
HEADS_011 = ("Q1_arrival", "Q2_sign", "Q3_magnitudes", "Q4_combined_ev")
BUDGETS_011 = tuple(f"{int(b * 100)}%" for b in D.BUDGETS)
N_PERM_011 = 2000                                    # prereg §5: >= 1000
PERM_SEED_011 = 20260828


def sign_flip_null(paired_by_unit: dict, n_perm: int = N_PERM_011,
                   seed: int = PERM_SEED_011) -> dict:
    """Window-level sign-flip permutation of per-unit paired differences.

    R-234, APPLIED AT CONSTRUCTION RATHER THAN REPAIRED LATER. The units are
    consumed in SORTED KEY ORDER, not dict/set order. A seed pins the RNG
    STREAM; it does not pin WHAT THE STREAM IS APPLIED TO. The predecessor
    instrument derived its order from a set of string keys under an unpinned
    PYTHONHASHSEED, so every run applied the same seeded sign sequence to a
    different data order and was an independent Monte-Carlo draw rather than a
    replay. Sorting the keys is the whole fix and it belongs here, at the point
    of consumption, where it cannot be undone by a caller's iteration order."""
    import random
    keys = sorted(paired_by_unit)                    # <- the fix
    vals = [float(paired_by_unit[k]) for k in keys]
    obs = math.fsum(vals)
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        t = math.fsum(v if rng.getrandbits(1) else -v for v in vals)
        if abs(t) >= abs(obs):
            ge += 1
    return {
        "observed": obs, "n_units": len(vals),
        "n_units_positive": sum(1 for v in vals if v > 0),
        "n_units_negative": sum(1 for v in vals if v < 0),
        "n_perm": n_perm, "perm_seed": seed,
        # (ge+1)/(n+1): the observed arrangement is itself one arrangement
        # under H0, so a permutation p can never be exactly zero.
        "p_two_sided": (ge + 1) / (n_perm + 1),
        "unit_order": "SORTED KEYS — pinned at consumption (R-234)",
    }


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni across the WHOLE family, monotonicity enforced."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        adj = max(adj, prev)                          # step-down monotonicity
        out[k] = adj
        prev = adj
    return out


def cell_key(arm: str, head: str, budget: str) -> str:
    return f"{arm}/{head}/{budget}"


def declared_family() -> dict:
    """The multiplicity family, DECLARED and counted before any result.

    R-232 9.1: 2 arms x 4 heads x 3 budgets = 24 cells, read JOINTLY. Counted
    whether or not a head fails — a four-head decomposition quadruples the
    family, and that cost must not be paid for by reporting only the head that
    won (prereg §5(4))."""
    cells = [cell_key(a, h, b) for a in ARMS_011
             for h in HEADS_011 for b in BUDGETS_011]
    return {"cells": cells, "n_cells": len(cells),
            "arms": list(ARMS_011), "heads": list(HEADS_011),
            "budgets": list(BUDGETS_011),
            "read": "JOINTLY, Holm-Bonferroni across the whole family",
            "counted_including_failing_heads": True}


def cluster_disclosure(G_complete_utc_days: int, unit_used: str) -> dict:
    """prereg §5(3). Below the ruled unit, say so IN the artifact."""
    ruled = "UTC day"
    return {
        "ruled_unit": ruled, "unit_used": unit_used,
        "G_complete_utc_days": G_complete_utc_days,
        "weaker_than_ruled": unit_used != ruled,
        "why": (f"G={G_complete_utc_days} complete UTC days, so the ruled unit "
                f"has too few replicates; {unit_used} is the finest plausibly-"
                f"exchangeable substitute. Units within a day are NOT "
                f"independent, so these p-values are OPTIMISTIC — evidence, "
                f"never a significance certificate."),
        "intervals_claimable": G_complete_utc_days >= 5,
    }


if __name__ == "__main__":
    raise SystemExit(selftest())
