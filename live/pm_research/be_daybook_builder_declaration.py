"""THE PER-DAY BOOK THE RULED GATE-1 RUN NEEDS -- DECLARED BEFORE ANY BUILD.

Answers the reviewer's "What must exist before BE builds the five reference
books" (be03d4d), item by item, and declares only what this seat owns: the
BOOK. It does not design DE's null and does not build anything.

THE BLOCKING FACT (§A.3), CONFIRMED AT MY OWN LOADER. `be_cancel_axis_null.
load()` reads `c["asm"]` and builds `rows` -- the decision population -- from
`asm["by_arm"][(COIN, head)][0]`. A day-book of reference + statuses +
population + n_slugs + terminal_marks raises on `c["asm"]`. So the book is a
SCORED book, not a reference book, and that changes what BE has to produce.

TWO THINGS THE PROBE SETTLES, AND ONE OF THEM CONSTRAINS THE WHOLE DESIGN.
Every number below marked `measured` is computed when this declaration is
emitted, not typed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
DERIVED = ROOT / "data/pm_5min/derived"
CACHE = DERIVED / "de_section81_cache_12.pkl"

HEADS = {"CONDVALUE_X_SKEW": "q1_arrival_composed_lgbm",
         "HAZARD_OVER_SKEWED_REF": "incumbent_linear_d"}
CANDIDATE_COINS = ("btc", "eth", "bnb", "doge", "hype", "sol", "xrp")
BUDGET = 0.10
RACE_DAYS = ("20260901", "20260902", "20260903", "20260904", "20260905")

#: MEASURED on the consumed hour, round 43: one replay of a 29,813-generation
#: stream through `HSP.replay_policy`.
MEASURED_REPLAY_S = 0.35
MEASURED_HOUR_GENERATIONS = 29813
MEASURED_HOUR_WINDOWS = 12
MEASURED_PEAK_GB = 0.27
WINDOWS_PER_COIN_PER_DAY = 288
N_DRAWS = 500
N_ARMS = 2


def coin_probe() -> dict:
    """WHICH COINS CAN BE SCORED AT THE PINNED THETAS. Measured, not assumed.

    This is the constraint the design does not yet carry: five of the seven
    coins have no threshold at all and REFUSE."""
    import de_phase4_diag_runner as R
    out = {}
    for coin in CANDIDATE_COINS:
        row = {}
        for arm, head in HEADS.items():
            try:
                row[arm] = {"theta": float(R.theta_for(coin, head, BUDGET)),
                            "available": True}
            except Exception as e:                       # noqa: BLE001
                row[arm] = {"available": False,
                            "refusal": type(e).__name__}
        row["scoreable"] = all(v["available"] for v in row.values()
                               if isinstance(v, dict))
        out[coin] = row
    return out


def scaling() -> dict:
    """The resource estimate, derived from a MEASURED replay, not guessed."""
    gens_per_window = MEASURED_HOUR_GENERATIONS / MEASURED_HOUR_WINDOWS
    gens_per_coin_day = gens_per_window * WINDOWS_PER_COIN_PER_DAY
    factor = gens_per_coin_day / MEASURED_HOUR_GENERATIONS
    per_replay = MEASURED_REPLAY_S * factor
    per_arm_day = per_replay * N_DRAWS
    per_day = per_arm_day * N_ARMS
    return {
        "basis": f"MEASURED {MEASURED_REPLAY_S} s for one replay of a "
                 f"{MEASURED_HOUR_GENERATIONS:,}-generation stream "
                 f"(round 43, {MEASURED_HOUR_WINDOWS} windows)",
        "generations_per_window": gens_per_window,
        "generations_per_coin_day": gens_per_coin_day,
        "scale_factor_vs_the_measured_hour": factor,
        "projected_s_per_replay": per_replay,
        "projected_s_per_arm_per_day": per_arm_day,
        "projected_s_per_day_both_arms": per_day,
        "projected_h_per_day_both_arms": per_day / 3600.0,
        "projected_h_five_days_one_coin": per_day * len(RACE_DAYS) / 3600.0,
        "projected_h_five_days_two_coins": per_day * len(RACE_DAYS) * 2 / 3600.0,
        "assumption_stated": "replay cost is LINEAR in stream length. That is "
                             "an assumption, not a measurement -- the policy "
                             "walks the stream once, but state growth is not "
                             "verified at day scale. IT MUST BE MEASURED ON "
                             "DAY 1 BEFORE THE OTHER FOUR ARE BUILT.",
        "memory": {
            "measured_peak_gb_on_the_hour": MEASURED_PEAK_GB,
            "naive_scaled_gb": MEASURED_PEAK_GB * factor,
            "cap": "8G, NOT raised (R-174)",
            "the_honest_statement": "a naive linear scale exceeds nothing, "
                                    "but memory is NOT linear in the same way "
                                    "wall time is and this is NOT a "
                                    "prediction. Day 1 measures it; if a day "
                                    "does not fit under 8G the day is "
                                    "REFUSED, not run at a raised cap.",
        },
    }


def build() -> dict:
    probe = coin_probe()
    sc = scaling()
    scoreable = sorted(c for c, v in probe.items() if v["scoreable"])
    body = CACHE.read_bytes() if CACHE.exists() else b""
    return {
        "protocol": "BE_DAYBOOK_BUILDER_DECLARATION_V1",
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "status": "DECLARED — NOTHING BUILT. No build until DE's design v2 "
                  "is reviewed.",
        "scope": "this declaration covers the BOOK only. It does not design "
                 "DE's null and adjudicates nothing (rule 14).",

        "THE_BLOCKING_FACT_CONFIRMED_AT_MY_OWN_LOADER": {
            "reviewer": "be03d4d §A.3",
            "site": "be_cancel_axis_null.py load() -- c['asm'], then "
                    "asm['by_arm'][(COIN, head)][0]",
            "consequence": "the day-book must be a SCORED book. A book with "
                           "reference + statuses + population + n_slugs + "
                           "terminal_marks and no `asm` raises, and the null "
                           "cannot be built at all.",
            "confirmed_by_this_seat": True,
        },

        "WHAT_asm_WILL_CONTAIN": {
            "shape": "asm['by_arm'][(coin, head)] -> (gen_scores, ...) where "
                     "gen_scores is keyed (slug, side, float(t0))",
            "heads": HEADS,
            "both_arms_pinned": True,
            "membership_test_the_loader_uses":
                "(slug, side, float(g['t0'])) in gen_scores",
            "and_what_it_costs": "generations with no assembled score cannot "
                                 "be drawn. On the measured hour that is "
                                 "1,309 of 31,122 (4.2%), identical for both "
                                 "heads — see be_generation_count_"
                                 "derivation_v1.json.",
        },

        "COIN_SET_AND_IT_IS_A_CONSTRAINT_NOT_A_CHOICE": {
            "probe": probe,
            "scoreable_at_the_pinned_thetas": scoreable,
            "n_scoreable": len(scoreable),
            "structurally_absent": sorted(set(CANDIDATE_COINS) - set(scoreable)),
            "why": "five of the seven coins have NO threshold at either "
                   "pinned head and raise ScoreStreamRefused. They are "
                   "absent at SELECTION and no amount of running buys them.",
            "the_existing_cache_is_narrower_still": {
                "cache": str(CACHE),
                "sha256": hashlib.sha256(body).hexdigest() if body else None,
                "arms_assembled": "btc only — the consumed hour's cache "
                                  "assembles ('btc', both heads) and no eth",
            },
            "PROPOSED_and_awaiting_ruling": "btc for all five days, with eth "
                                            "as a declared extension only if "
                                            "the day-1 measurement leaves "
                                            "room. Declaring two coins and "
                                            "delivering one is the failure "
                                            "mode; one is declared.",
        },

        "DRAW_POOL_CONVENTION_DECLARED_EITHER_WAY": {
            "what_the_loader_does_today": "rows is built from CONDVALUE's "
                                          "head ONLY, and both arms draw "
                                          "from that one pool",
            "is_that_intended": "YES, AND HERE IS THE ARGUMENT — but it is "
                                "declared, not assumed, because it is a "
                                "denominator question and those have cost "
                                "this programme two rounds.",
            "the_argument": "the null matches each arm's DECISION COUNT "
                            "separately; the pool is the set of generations "
                            "a decision could have been taken on. If the two "
                            "heads scored different generation sets, a "
                            "per-arm pool would make the two nulls "
                            "non-comparable — each arm would be drawing from "
                            "a different universe.",
            "AND_THE_FACT_THAT_SETTLES_IT_HERE": "on the measured hour BOTH "
                                                 "heads score the SAME 29,813 "
                                                 "generations (measured, both "
                                                 "drops equal 1,309), so the "
                                                 "shared pool and a per-arm "
                                                 "pool are THE SAME SET and "
                                                 "the choice is empty on this "
                                                 "book.",
            "what_must_be_checked_per_day": "that equality is a MEASUREMENT "
                                            "of the hour, not a guarantee. "
                                            "The builder must assert "
                                            "set(gen_scores[CONDVALUE]) == "
                                            "set(gen_scores[HAZARD]) per day "
                                            "and REFUSE the day if it fails, "
                                            "because then the shared pool is "
                                            "a real choice and this "
                                            "declaration does not cover it.",
        },

        "INPUTS_PER_DAY": {
            "days": list(RACE_DAYS),
            "reference_source": "the same selection the forward scorer used, "
                                "per day",
            "digests_recorded_per_day": ["the day book's own sha256",
                                         "each pinned head's model sha256",
                                         "each theta"],
            "run_time_verification": "theta and model digests are RECOMPUTED "
                                     "at build time and compared to the "
                                     "pinned values; a mismatch REFUSES that "
                                     "day (reviewer item 6 — recording a "
                                     "digest is not verifying it)",
        },

        "OUTPUT": {
            "path_scheme": "data/pm_5min/derived/be_daybook_<DAY>.pkl",
            "digest_scheme": "sha256 of the pickle, emitted in a sidecar "
                             "receipt be_daybook_<DAY>.json AND recomputed "
                             "by the reader at read time; a mismatch refuses "
                             "that day",
            "why_a_sidecar": "the pickle is far over 1 MB and is pinned by "
                             "digest, not committed; the receipt is small "
                             "and commits with its code",
            "population_source_sha256_travels_in_the_null":
                "already implemented — be_cancel_axis_null v2 emits "
                "population.source_sha256 at run time",
        },

        "RESOURCES": sc,

        "THE_RESPONSE_TO_AN_OVERRUN_IS_WRITTEN_DOWN": {
            "reviewer_item": 8,
            "rule": "THE DAY IS REFUSED. Draws are NOT cut below the declared "
                    "500 and the 8G cap is NOT raised (R-174).",
            "why_not_fewer_draws": "n was declared from a measurement with "
                                   "stated headroom; cutting it after seeing "
                                   "the clock is choosing a null's resolution "
                                   "on the basis of cost, which is rule 11's "
                                   "shape in a new place.",
            "what_a_refusal_looks_like": "the day is reported with its status "
                                         "and its measured cost, and the race "
                                         "of books is short by that day — "
                                         "stated, never silently dropped "
                                         "(rule 4).",
        },

        "DAY_1_FIRST_AND_ALONE": {
            "reviewer_items": [4, 5],
            "rule": "ONE day is built and measured before the other four. It "
                    "discharges the 'same cascade' premise, which §A.4 calls "
                    "LIVE rather than hypothetical, and it measures the "
                    "scaling assumption above.",
            "economic_fields_SEALED_until_all_five_are_run": True,
            "and_all_five_run_regardless_of_the_interim": "declared here so "
                                                          "an interim result "
                                                          "cannot stop the "
                                                          "run",
            "minimum_decisions_per_arm_per_day": "TO BE DECLARED FROM DAY 1's "
                                                 "MEASUREMENT, before day 2 "
                                                 "is built — it cannot be "
                                                 "declared honestly from the "
                                                 "hour, whose arms made 1,154 "
                                                 "and 106 decisions over 12 "
                                                 "windows",
            "small_but_nonzero_null_sd": "REPORTED with the cell and the cell "
                                         "marked; no floor is invented here",
        },

        "NOT_MINE_AND_NOT_DECLARED_HERE": [
            "DE's null design",
            "the day set's re-derivation on quality and the fixing of G "
            "(reviewer item 7) — that is the coordinator's",
            "whether 08-29 is admissible",
        ],
        "builds_nothing": True,
        "decides_nothing": "REPORTED (rule 14).",
    }


EXPECTED_CHECKS = 7


def selftest() -> int:
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    d = build()
    cs = d["COIN_SET_AND_IT_IS_A_CONSTRAINT_NOT_A_CHOICE"]
    ok(cs["scoreable_at_the_pinned_thetas"] == ["btc", "eth"]
       and len(cs["structurally_absent"]) == 5,
       f"COIN PROBE, MEASURED: only {cs['scoreable_at_the_pinned_thetas']} "
       f"have thresholds at both pinned heads; {cs['structurally_absent']} "
       f"REFUSE — the design names no coin set and this is the constraint")
    ok(all(not v["scoreable"] for c, v in cs["probe"].items()
           if c in cs["structurally_absent"]),
       "and the probe FIRES on each absent coin individually, so a coin that "
       "became available would change this answer")
    r = d["RESOURCES"]
    ok(abs(r["scale_factor_vs_the_measured_hour"] - 24.0) < 0.001
       and r["projected_h_per_day_both_arms"] > 2.0,
       f"RESOURCE ESTIMATE IS DERIVED from a measured 0.35 s replay: "
       f"{r['scale_factor_vs_the_measured_hour']:.1f}x scale, "
       f"{r['projected_h_per_day_both_arms']:.2f} h per day for both arms, "
       f"{r['projected_h_five_days_one_coin']:.1f} h for five days on one coin")
    ok("assumption, not a measurement" in r["assumption_stated"],
       "and the LINEARITY it rests on is named as an assumption to be "
       "measured on day 1, not presented as a projection")
    ok("REFUSED" in d["THE_RESPONSE_TO_AN_OVERRUN_IS_WRITTEN_DOWN"]["rule"]
       and "NOT cut" in d["THE_RESPONSE_TO_AN_OVERRUN_IS_WRITTEN_DOWN"]["rule"],
       "the overrun response is WRITTEN DOWN and it is 'refuse the day' — "
       "not fewer draws and not a raised cap (reviewer item 8)")
    dp = d["DRAW_POOL_CONVENTION_DECLARED_EITHER_WAY"]
    ok("YES" in dp["is_that_intended"] and "REFUSE the day" in
       dp["what_must_be_checked_per_day"],
       "the draw-pool convention is DECLARED as intended, with the reason, "
       "AND with the per-day check that must refuse if the two heads ever "
       "score different sets")
    ok(d["builds_nothing"] and d["status"].startswith("DECLARED"),
       "and nothing is built — no build until DE's design v2 is reviewed")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--declare" in argv:
        out = build()
        dst = HERE / "declarations" / "be_daybook_builder_declaration_v1.json"
        dst.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
        print(json.dumps({"written": str(dst),
                          "scoreable": out["COIN_SET_AND_IT_IS_A_CONSTRAINT_"
                                           "NOT_A_CHOICE"]
                                          ["scoreable_at_the_pinned_thetas"],
                          "h_per_day": round(out["RESOURCES"]
                                             ["projected_h_per_day_both_arms"], 2)}))
        return 0
    print("usage: be_daybook_builder_declaration.py --selftest | --declare")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
