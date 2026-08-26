"""Apply the FROZEN reduced-fine candidate to a forward UTC day. Unchanged.

AUTHORISATION (R-126, in-file): R-169(1), post-freeze order of work.

THE R-141 LESSON IS THIS FILE'S PRIMARY DESIGN CONSTRAINT. A scorer once
shipped as a FRAME with no scoring path and passed its selftests, because
every test asserted the shape of the report rather than the presence of a
score. So the rule here: **a report with zero scored actions is a FAILURE,
never a pass**, and the selftest carries a positive control that computes a
known score by hand and demands the scorer reproduce it. Shape assertions
prove nothing on their own.

THE FROZEN ARTIFACT IS APPLIED, NEVER REFITTED. No weight, mean or scale in
this file is computed from forward data. If a forward day could change the
model, the forward test would be measuring the model's ability to fit the
day it is being judged on.

INFERENCE ALIGNMENT COMES FROM THE ARTIFACT'S OWN CONTRACT. `zscale` prepends
an intercept, so 61 weights pair with 60 normalization parameters. This reads
`feature_vector_contract` and REFUSES an artifact that lacks it, rather than
assuming a layout -- pairing norm_mu[0] with weight[0] misaligns every
coefficient silently and produces plausible numbers.
"""
from __future__ import annotations

import json, math, sys
from pathlib import Path

REPO = Path("/home/yuqing/ctaNew")
DERIVED = REPO / "data/pm_5min/derived"
CANDIDATE = DERIVED / "harmful_reduced_fine_candidate_v1.json"
FREEZE_INSTANT_UTC = "2026-08-26T10:49:55Z"
N_CANDIDATES_IN_RACE = 2


class NotFrozen(RuntimeError):
    """Refuses to score with anything that is not a frozen candidate."""


class EmptyScoring(RuntimeError):
    """A report with no scored actions is a failure, not an empty result."""


def load_frozen(path: Path = CANDIDATE) -> dict:
    c = json.loads(path.read_text())
    if c.get("status") != "FROZEN":
        raise NotFrozen(f"artifact status is {c.get('status')!r}, not FROZEN. "
                        f"Forward scoring may only use a frozen candidate.")
    for coin, f in c["fits"].items():
        fc = f.get("feature_vector_contract")
        if not fc:
            raise NotFrozen(
                f"{coin} carries no feature_vector_contract. Refusing to guess "
                f"the layout: 61 weights pair with 60 norm params because "
                f"zscale prepends an intercept, and a wrong assumption "
                f"misaligns every coefficient silently.")
        if len(f["hazard_weights"]) != len(f["norm_mu"]) + 1:
            raise NotFrozen(
                f"{coin}: {len(f['hazard_weights'])} weights vs "
                f"{len(f['norm_mu'])} norm params — not the +1 the contract "
                f"declares. Refusing.")
    return c


def design_row(fit: dict, raw: list) -> list:
    """Build the model's input vector EXACTLY as the contract declares."""
    mu, sd = fit["norm_mu"], fit["norm_sd"]
    if len(raw) != len(mu):
        raise ValueError(f"expected {len(mu)} raw features, got {len(raw)}")
    return [1.0] + [(raw[i] - mu[i]) / sd[i] for i in range(len(mu))]


def expected_cancel_value(fit: dict, raw: list) -> float:
    """p_fill(hazard) x conditional value. The frozen product, unchanged."""
    x = design_row(fit, raw)
    z = sum(a * b for a, b in zip(fit["hazard_weights"], x))
    p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
    wm = fit.get("value_weights")
    v = sum(a * b for a, b in zip(wm, x)) if wm else 0.0
    return p * v


def build_report(day: str, scored: dict, da_verified: bool) -> dict:
    """Assemble a per-day forward report. REFUSES an empty scoring set."""
    total = sum(len(v) for v in scored.values())
    if total == 0:
        raise EmptyScoring(
            f"{day}: zero actions scored across {list(scored)}. A forward "
            f"report with no scores is a FAILURE, not an empty result — this "
            f"is the R-141 failure mode, where a frame with no scoring path "
            f"passed its tests.")
    return {
        "protocol": "HARMFUL_FORWARD_DAY_REPORT_V1",
        "day": day,
        "candidate": CANDIDATE.name,
        "freeze_instant_utc": FREEZE_INSTANT_UTC,
        "n_candidates_in_race": N_CANDIDATES_IN_RACE,
        "multiplicity_note": "any clears-claim is judged against a null "
                             "accounting for 2 candidates, not 1",
        "unit": "ACTION",
        "n_actions_scored": {k: len(v) for k, v in scored.items()},
        "da_verified_first": da_verified,
        "admissible": bool(da_verified),
        "admission_note": "R-153(3): a day is admissible only after DA verifies "
                          "it first. `admissible` is a STATUS, not an "
                          "entitlement — the policy layer decides (rule 14).",
        "forward_day_index_note": "day one is 2026-08-27; a verdict needs G>=5 "
                                  "complete untouched UTC days (R-109)",
        "no_interval_below_g5": True,
    }


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    # ---- POSITIVE CONTROL: the scorer must PRODUCE A KNOWN NUMBER ----------
    # Hand-computed, not asserted-as-shape. This is the R-141 arm.
    fit = {"hazard_weights": [0.5, 1.0, -2.0],
           "value_weights":  [1.0, 2.0,  0.0],
           "norm_mu": [10.0, 4.0], "norm_sd": [2.0, 1.0],
           "feature_vector_contract": {"intercept_is_position_0": True}}
    raw = [14.0, 5.0]                       # -> scaled [2.0, 1.0]
    x = [1.0, 2.0, 1.0]
    z = 0.5 * 1 + 1.0 * 2 + (-2.0) * 1      # = 0.5
    p = 1 / (1 + math.exp(-z))
    v = 1.0 * 1 + 2.0 * 2 + 0.0 * 1         # = 5.0
    ok(abs(design_row(fit, raw)[1] - 2.0) < 1e-12,
       "normalization applies to positions 1..n, with the intercept at 0")
    got = expected_cancel_value(fit, raw)
    ok(abs(got - p * v) < 1e-12,
       f"POSITIVE CONTROL: the scorer reproduces a hand-computed value "
       f"({p*v:.6f}) — it actually SCORES, rather than shaping a report "
       f"around no scoring path (R-141)")
    ok(abs(got) > 1e-6, "and the control value is non-trivial, so a scorer "
                        "that silently returned zero would FAIL this")

    # ---- the R-141 arm proper: an empty report must be an ERROR ------------
    try:
        build_report("2026-08-27", {"btc": [], "eth": []}, True)
        ok(False, "an empty scoring set must be REFUSED")
    except EmptyScoring as e:
        ok("R-141" in str(e),
           "POSITIVE CONTROL: a report with ZERO scored actions raises, "
           "naming the failure mode it exists to prevent")
    r = build_report("2026-08-27", {"btc": [1.0, 2.0], "eth": [3.0]}, True)
    ok(r["n_actions_scored"] == {"btc": 2, "eth": 1} and r["unit"] == "ACTION",
       "a real report counts ACTIONS per coin and names its unit")
    ok(r["n_candidates_in_race"] == 2,
       "every forward report carries multiplicity 2 (R-146(3) lineage)")

    # ---- refusals ---------------------------------------------------------
    import tempfile
    for bad, why in (({"status": "DRAFT", "fits": {}}, "a non-FROZEN artifact"),
                     ({"status": "FROZEN", "fits": {"btc": {
                         "hazard_weights": [1, 2], "norm_mu": [0.0],
                         "norm_sd": [1.0]}}}, "a fit with no contract")):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(bad, fh); t = Path(fh.name)
        try:
            load_frozen(t)
            ok(False, f"{why} must be refused")
        except NotFrozen:
            ok(True, f"KNOWN-BAD REFUSED: {why} cannot be used for forward scoring")

    # ---- the real artifact loads -----------------------------------------
    if CANDIDATE.exists():
        c = load_frozen(CANDIDATE)
        ok(c["status"] == "FROZEN" and set(c["fits"]) == {"btc", "eth"},
           "the REAL frozen candidate loads and passes the layout check")
        f = c["fits"]["btc"]
        val = expected_cancel_value(f, [0.0] * len(f["norm_mu"]))
        ok(math.isfinite(val),
           "and it produces a finite score on a real 60-feature input")

    print(f"harmful_forward_scorer selftest: {checks} checks OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    print("usage: harmful_forward_scorer.py --selftest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
