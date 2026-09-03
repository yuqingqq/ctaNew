"""Synthetic-day dry run for the forward scorer. The R-141 lesson, executable.

AUTHORISATION (R-126, in-file): R-175(2), post-freeze queue.

WHY THIS EXISTS AND WHY IT IS NOT A SELFTEST. `harmful_forward_scorer` has
unit tests that pass. So did the scorer that shipped as a frame with no scoring
path. Unit tests prove the PIECES work; this proves the PATH works -- a
fabricated day goes in at one end and real per-action scores come out the
other, through the same code day one will use.

WHAT IS FABRICATED AND WHAT IS NOT. The exposure ROWS are synthetic, because
2026-08-27 does not exist yet. The MODEL is the real frozen artifact, loaded
from disk, applied through its own feature_vector_contract. So the dry run
exercises the real inference path against a known-shaped day.

THE DRY RUN FAILS LOUDLY IN THREE WAYS, each a thing that would otherwise be
discovered on day one with a day of tape at stake:
  * the scorer produces no scores at all  (the R-141 frame)
  * it produces constant scores           (a dead model reads as a live one)
  * the report is emitted without an action count or with a zero one
"""
from __future__ import annotations

import json, math, random, sys
from pathlib import Path

sys.path.insert(0, "/home/yuqing/ctaNew/live/pm_research")
import harmful_forward_scorer as fs

DAY = "2099-01-01"          # deliberately absurd: cannot collide with real tape
N_WINDOWS, N_GENS, ROWS_PER_GEN = 12, 40, 3


def synth_rows(coin: str, n_feat: int, seed: int) -> list:
    rnd = random.Random(seed)
    rows = []
    for w in range(N_WINDOWS):
        t0 = 4070908800 + w * 300           # 2099-01-01, 5-min slugs
        slug = f"{coin}-updown-5m-{t0}"
        for g in range(N_GENS):
            side = "BUY_UP" if (g % 2 == 0) else "SELL_UP"
            for k in range(ROWS_PER_GEN):
                pv = rnd.gauss(0, 40) if rnd.random() < 0.25 else 0.0
                rows.append({
                    "slug": slug, "coin": coin, "day": DAY, "t0": t0,
                    "t_start": -300.0 + g * 6.0 + k * 0.4, "side": side,
                    "gen": g, "status": "OK",
                    "raw": [rnd.gauss(0, 1) for _ in range(n_feat)],
                    "any_fill_ahead": pv != 0.0,
                    "latency": {"50": {"preventable_value_cents": pv,
                                       "preventable_shares": 1.0 if pv else 0.0,
                                       "stale_shares": 0.0}},
                })
    return rows


def main() -> int:
    import harmful_action_eval as ae
    cand = fs.load_frozen(expect=fs.declared_candidate_identity())
    print(f"  frozen candidate loaded: {fs.CANDIDATE.name}, "
          f"frozen_at {cand['frozen_at_utc']}")
    failures = []
    report_in = {}
    for coin in ("btc", "eth"):
        fit = cand["fits"][coin]
        n_feat = len(fit["norm_mu"])
        rows = synth_rows(coin, n_feat, seed=7 if coin == "btc" else 11)
        scores = [fs.expected_cancel_value(fit, r["raw"]) for r in rows]

        # FAILURE 1 -- no scores at all (the R-141 frame)
        if not scores:
            failures.append(f"{coin}: scorer produced NO scores")
            continue
        # FAILURE 2 -- constant scores (a dead model looks alive in aggregate)
        spread = max(scores) - min(scores)
        if spread < 1e-9:
            failures.append(f"{coin}: scores are CONSTANT (spread {spread:.2e}) "
                            f"-- a dead model would pass a shape check")
        if not all(math.isfinite(s) for s in scores):
            failures.append(f"{coin}: non-finite scores present")

        gate = ae.evaluate_policy(rows, scores, latency_ms=50,
                                  budgets=(0.05, 0.10, 0.15), n_random=200)
        n_act = gate.get("n_actions")
        # FAILURE 3 -- a report with no actions
        if not n_act:
            failures.append(f"{coin}: gate reported n_actions={n_act!r}")
        report_in[coin] = [1.0] * (n_act or 0)
        print(f"  {coin}: {len(rows)} rows -> {n_act} actions | "
              f"score spread {spread:.4f} | "
              f"net@10% {gate['budgets']['10%']['net_cents']:+.1f}c | "
              f"hours {gate['budgets']['10%']['concentration']['n_hours_with_cancellations']}")

    rep = fs.build_report(DAY, report_in, da_verified=True)
    print(f"\n  report: unit={rep['unit']} candidates={rep['n_candidates_in_race']} "
          f"actions={rep['n_actions_scored']} admissible={rep['admissible']}")

    # the dry run must itself be falsifiable: an EMPTY day must be refused
    try:
        fs.build_report(DAY, {"btc": [], "eth": []}, da_verified=True)
        failures.append("an empty day was ACCEPTED by build_report")
    except fs.EmptyScoring:
        print("  control: an empty day is REFUSED (R-141 arm fires)")

    if failures:
        print("\nDRY RUN FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nDRY RUN PASSED — the scoring PATH fires end-to-end on a day-shaped "
          "input, with the real frozen model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
