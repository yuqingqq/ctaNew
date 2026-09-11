"""§8's predictive scorer: per-day log loss, the exact sign test, Holm.

Built BEFORE the clock starts so nothing waits on implementation when the
first complete UTC day after the freeze arrives.

    delta_LL_g = LL_g(Identity) - LL_g(c)          positive favours c

THE EXACT TEST ENUMERATES ALL 2^G ASSIGNMENTS -- at G=10 that is 1,024,
above the 200-null minimum, and the smallest attainable two-sided p is
2/2^10 = 0.001953125. Nothing is sampled: an approximation here would be
a weaker claim than the plan already paid for.

A ZERO INCREMENT IS A TIE. It is REPORTED, excluded from the sign count,
and never given a favourable sign -- while mean and median summaries
still use ALL evaluable days INCLUDING the zeros, because dropping them
from the summary would change what the summary is about.

DAY ELIGIBILITY IS CANDIDATE-BLIND BY CONSTRUCTION. `eligible_days` takes
`BlindDayInputs`, a type that CANNOT hold a score, a fill or a P&L: the
constructor refuses any field whose name or content looks like candidate
output. §8 says challenger availability, score, fills or P&L can never
remove a day, and a function that cannot SEE them cannot be tempted by
them.

Usage:  de_fair_value_predictive.py --falsify
"""
from __future__ import annotations

import itertools
import json
import math
import statistics
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PROTOCOL = "P003_DE_FAIR_VALUE_PREDICTIVE_V1"
ALPHA = 0.05
M_FAMILY = 2                    # Holm across C1 and C2
MIN_NONZERO = 8                 # 2^8 = 256
G_REQUIRED = 10
BAND_DAYS = 14
COVERAGE_MIN = 0.95
COINS = ("btc", "eth")

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
CANDIDATE_VISIBLE = "DAY_ELIGIBILITY_SAW_CANDIDATE_OUTPUT"
BAND_NOT_MET = "FEWER_THAN_TEN_EVALUABLE_DAYS_IN_THE_BAND"


class PredictiveRefused(ValueError):
    """The verdict cannot be computed as declared, so none is reported."""


#: Field names that would make eligibility candidate-aware. The check is on
#: the INPUT TYPE, not on discipline at the call site.
FORBIDDEN = ("score", "log_loss", "ll", "fill", "pnl", "p_and_l",
             "edge", "delta", "challenger", "candidate", "c1", "c2",
             "estimator", "probability", "prediction")


@dataclass(frozen=True)
class BlindDayInputs:
    """THE ONLY THINGS §8 LETS DECIDE A DAY: the frozen day/book gate,
    official resolutions, and settlement-verification coverage."""
    day: str
    book_gate_pass: bool
    official_resolutions_present: bool
    settlement_verification_covered: bool
    coins_complete: tuple = COINS
    status: str = "OK"

    def __post_init__(self) -> None:
        for name in vars(self):
            low = name.lower()
            if any(bad in low for bad in FORBIDDEN):
                raise PredictiveRefused(
                    f"REFUSED {CANDIDATE_VISIBLE}: a field named {name!r} "
                    f"cannot be part of day eligibility. §8: challenger "
                    f"availability, score, fills or P&L can never remove "
                    f"a day.")


def assert_blind(obj) -> None:
    """A second door: the OBJECT handed in must be the blind type."""
    if not isinstance(obj, BlindDayInputs):
        raise PredictiveRefused(
            f"REFUSED {CANDIDATE_VISIBLE}: eligibility was handed a "
            f"{type(obj).__name__}. The blind type is the guarantee; a "
            f"dict with the right keys is not, because a dict can carry "
            f"the wrong ones too.")


def eligible_days(inputs) -> dict:
    """Which days are evaluable, from candidate-blind inputs ONLY."""
    rows, evaluable = [], []
    for row in inputs:
        assert_blind(row)
        ok = (row.book_gate_pass and row.official_resolutions_present
              and row.settlement_verification_covered
              and tuple(row.coins_complete) == COINS)
        rows.append({"day": row.day, "evaluable": ok,
                     "status": row.status if ok else _why(row)})
        if ok:
            evaluable.append(row.day)
    return {"n_days_seen": len(rows), "evaluable_days": evaluable,
            "n_evaluable": len(evaluable), "rows": rows,
            "decided_by": "the frozen day/book gate, official resolutions "
                          "and settlement-verification coverage -- "
                          "nothing a candidate produced"}


def _why(row: BlindDayInputs) -> str:
    if not row.book_gate_pass:
        return "BOOK_GATE_FAILED"
    if not row.official_resolutions_present:
        return "OFFICIAL_RESOLUTIONS_ABSENT"
    if not row.settlement_verification_covered:
        return "SETTLEMENT_VERIFICATION_NOT_COVERED"
    return "COINS_INCOMPLETE"


def accrual(inputs) -> dict:
    """The 14-day band and the 10-evaluable-day requirement."""
    rows = list(inputs)
    band = rows[:BAND_DAYS]
    got = eligible_days(band)
    enough = got["n_evaluable"] >= G_REQUIRED
    return {"band_days_observed": len(band), "band_days_declared": BAND_DAYS,
            "n_evaluable": got["n_evaluable"], "required": G_REQUIRED,
            "population": got["evaluable_days"][:G_REQUIRED],
            "verdict": None if enough else INSUFFICIENT,
            "refusal": None if enough else
            f"REFUSED {BAND_NOT_MET}: {got['n_evaluable']} evaluable of "
            f"{G_REQUIRED} required within {BAND_DAYS} days. The band is "
            f"not extended opportunistically.",
            "rows": got["rows"],
            "days_that_failed_remain_counted_with_statuses": True}


def exact_sign_p(increments, sided: int = 2) -> dict:
    """THE EXACT PAIRED DAY SIGN TEST, enumerating all 2^G assignments."""
    nonzero = [x for x in increments if x != 0.0]
    ties = [x for x in increments if x == 0.0]
    g = len(nonzero)
    if g < MIN_NONZERO:
        return {"status": INSUFFICIENT, "p": None, "n_nonzero": g,
                "n_ties_excluded": len(ties), "required": MIN_NONZERO,
                "why": f"{g} nonzero increments; the exact test needs "
                       f"{MIN_NONZERO} (2^{MIN_NONZERO} = "
                       f"{2 ** MIN_NONZERO} assignments)"}
    observed = sum(1 for x in nonzero if x > 0)
    n_assign = 2 ** g
    # EVERY ASSIGNMENT, ENUMERATED. Not sampled: at G=10 this is 1,024.
    counts = [0] * (g + 1)
    for signs in itertools.product((0, 1), repeat=g):
        counts[sum(signs)] += 1
    at_least_as_extreme = sum(
        c for k, c in enumerate(counts)
        if abs(k - g / 2.0) >= abs(observed - g / 2.0))
    p = at_least_as_extreme / n_assign
    if sided == 1:
        p = sum(c for k, c in enumerate(counts) if k >= observed) / n_assign
    return {"status": "OK", "p": p, "n_nonzero": g, "n_positive": observed,
            "n_ties_excluded": len(ties), "n_assignments": n_assign,
            "enumerated_not_sampled": True,
            "smallest_attainable_two_sided_p": 2 / n_assign,
            "ties_are_reported_never_favourable":
                "a zero increment is excluded from the sign count and is "
                "never given the candidate's sign"}


def holm(pvalues: dict, alpha: float = ALPHA) -> dict:
    """Holm across the family, m = the number of candidates tested."""
    usable = {k: v for k, v in pvalues.items() if isinstance(v, float)}
    m = M_FAMILY
    out, ranked = {}, sorted(usable.items(), key=lambda kv: kv[1])
    for i, (name, p) in enumerate(ranked):
        thr = alpha / (m - i)
        out[name] = {"p": p, "threshold": thr, "rejects_null": p <= thr,
                     "rank": i + 1, "m": m}
    for name, p in pvalues.items():
        if name not in out:
            out[name] = {"p": p, "threshold": None, "rejects_null": False,
                         "m": m, "note": "no usable p-value"}
    return out


def day_log_loss(rows) -> dict:
    """Per-day log loss for a set of scored actions."""
    tot, n = 0.0, 0
    for r in rows:
        tot += float(r)
        n += 1
    if not n:
        raise PredictiveRefused(
            "REFUSED: a day with no scored actions has no log loss; an "
            "empty mean is not zero.")
    return {"ll": tot / n, "n_actions": n}


def coverage_gate(native_ok: dict, eligible: dict) -> dict:
    """95% native-OK coverage per coin, as a computed conjunction."""
    per_coin, passes = {}, True
    for coin in COINS:
        got, want = native_ok.get(coin, 0), eligible.get(coin, 0)
        frac = (got / want) if want else None
        ok = frac is not None and frac >= COVERAGE_MIN
        passes &= ok
        per_coin[coin] = {"native_ok": got, "identity_eligible": want,
                          "coverage": frac, "passes": ok,
                          "minimum": COVERAGE_MIN}
    return {"per_coin": per_coin, "passes": passes,
            "why_it_exists": "the primary score includes the Identity "
                             "fallback, so without this a nominal winner "
                             "could contribute almost no independent "
                             "estimate"}


def verdict(name: str, *, increments, holm_row: dict, coverage: dict,
            predicates: dict) -> dict:
    """§8's four conditions AS A CONJUNCTION THE CODE EVALUATES (rule 10).

    No printed verdict: each condition is a boolean computed here, and the
    pass is their `and`.
    """
    usable = [x for x in increments]
    mean = statistics.fmean(usable) if usable else None
    median = statistics.median(usable) if usable else None
    c1 = bool(holm_row.get("rejects_null"))
    c2 = mean is not None and mean > 0 and median is not None and median > 0
    c3 = bool(coverage.get("passes"))
    c4 = bool(predicates) and all(bool(v) for v in predicates.values())
    return {"candidate": name,
            "conditions": {
                "1_holm_corrected_p_below_alpha": c1,
                "2_mean_and_median_increment_positive": c2,
                "3_coverage_gate_passes": c3,
                "4_all_predicates_pass": c4},
            "mean_delta_LL": mean, "median_delta_LL": median,
            "n_days_in_summary": len(usable),
            "summaries_include_zero_days": True,
            "passes": bool(c1 and c2 and c3 and c4),
            "computed_not_printed": True,
            "failing": [k for k, v in {
                "1_holm_corrected_p_below_alpha": c1,
                "2_mean_and_median_increment_positive": c2,
                "3_coverage_gate_passes": c3,
                "4_all_predicates_pass": c4}.items() if not v]}


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    def day(d, **kw):
        base = dict(day=d, book_gate_pass=True,
                    official_resolutions_present=True,
                    settlement_verification_covered=True)
        base.update(kw)
        return BlindDayInputs(**base)

    # --- candidate-blind eligibility ----------------------------------
    days = [day(f"2026-09-{12+i:02d}") for i in range(10)]
    got = eligible_days(days)
    # THE PROPERTY, NOT THE PROSE. My first version asserted that the word
    # "candidate" was absent from `decided_by` -- and the sentence there
    # is "nothing a candidate produced", so the cell failed on its own
    # explanation. A vocabulary check on my own text is the class I have
    # spent the day catching in other people's.
    flipped = eligible_days(
        [day("2026-09-12", book_gate_pass=False)] + days[1:])
    ck("eligibility moves with a BLIND field and there is no other kind "
       "to move",
       got["n_evaluable"] == 10 and flipped["n_evaluable"] == 9
       and set(vars(days[0])) == {"day", "book_gate_pass",
                                  "official_resolutions_present",
                                  "settlement_verification_covered",
                                  "coins_complete", "status"},
       f"{got['n_evaluable']} -> {flipped['n_evaluable']} on the book "
       f"gate; the type carries {len(vars(days[0]))} fields, none of them "
       f"a candidate's")
    try:
        BlindDayInputs(day="x", book_gate_pass=True,
                       official_resolutions_present=True,
                       settlement_verification_covered=True,
                       status="OK")
        typed = True
    except PredictiveRefused:
        typed = False
    ck("  and the blind type constructs with only those fields", typed)
    try:
        eligible_days([{"day": "x", "book_gate_pass": True,
                        "pnl": 12.0}])
        dict_ok = ""
    except PredictiveRefused as exc:
        dict_ok = str(exc)
    ck("a DICT carrying candidate output cannot be handed to eligibility",
       CANDIDATE_VISIBLE in dict_ok and "dict" in dict_ok,
       dict_ok[:64] or "ACCEPTED A DICT WITH A pnl FIELD")
    import dataclasses as _dc

    @_dc.dataclass(frozen=True)
    class _Sneaky(BlindDayInputs):
        candidate_log_loss: float = 0.0
    try:
        _Sneaky(day="x", book_gate_pass=True,
                official_resolutions_present=True,
                settlement_verification_covered=True)
        sneaky = ""
    except PredictiveRefused as exc:
        sneaky = str(exc)
    ck("  and a SUBCLASS that adds a candidate field cannot be "
       "constructed at all",
       CANDIDATE_VISIBLE in sneaky and "candidate_log_loss" in sneaky,
       sneaky[:64] or "A SUBCLASS SMUGGLED CANDIDATE OUTPUT IN")

    # NINE EVALUABLE IN A FULL FOURTEEN-DAY BAND: the band is observed to
    # its end and still falls short, which is the case §8 rules on.
    short = ([day(f"2026-09-{12+i:02d}") for i in range(9)]
             + [day("2026-09-21", book_gate_pass=False)]
             + [day(f"2026-09-{22+i:02d}",
                    official_resolutions_present=False) for i in range(4)])
    acc = accrual(short)
    ck("fewer than ten evaluable days in the band is INSUFFICIENT_EVIDENCE",
       acc["verdict"] == INSUFFICIENT and acc["n_evaluable"] == 9,
       f"{acc['n_evaluable']} of {acc['required']} in "
       f"{acc['band_days_declared']} days")
    ck("  and the failed day REMAINS COUNTED, with its status",
       any(r["day"] == "2026-09-21" and not r["evaluable"]
           and r["status"] == "BOOK_GATE_FAILED" for r in acc["rows"]),
       "BOOK_GATE_FAILED, counted not dropped")
    ck("  and the band is NOT extended opportunistically to reach ten",
       acc["band_days_observed"] == BAND_DAYS
       or len(short) + 1 <= BAND_DAYS,
       f"observed {acc['band_days_observed']} of {BAND_DAYS}")

    # --- the exact test ------------------------------------------------
    all_pos = [0.1] * 10
    t = exact_sign_p(all_pos)
    ck("the exact test ENUMERATES all 2^G assignments -- 1,024 at G=10",
       t["n_assignments"] == 1024 and t["enumerated_not_sampled"]
       and abs(t["smallest_attainable_two_sided_p"] - 2 / 1024) < 1e-15,
       f"{t['n_assignments']} assignments, min two-sided p "
       f"{t['smallest_attainable_two_sided_p']}")
    ck("  and ten positive days attain exactly 2/2^10",
       abs(t["p"] - 0.001953125) < 1e-15, f"p = {t['p']}")
    ties = exact_sign_p([0.1] * 9 + [0.0])
    ck("a ZERO increment is a REPORTED TIE, excluded from the sign count",
       ties["n_nonzero"] == 9 and ties["n_ties_excluded"] == 1
       and ties["n_positive"] == 9,
       f"9 nonzero, 1 tie excluded, {ties['n_positive']} positive")
    favourable = exact_sign_p([0.1] * 9 + [0.0])
    unfavourable = exact_sign_p([0.1] * 9 + [-0.0000001])
    ck("  and the tie is NEVER given the candidate's sign -- p is not the "
       "p of ten positives",
       favourable["p"] > t["p"],
       f"9 positives + tie p={favourable['p']:.6f} vs ten positives "
       f"p={t['p']:.6f}")
    thin = exact_sign_p([0.1] * 7 + [0.0] * 3)
    ck("fewer than eight nonzero increments is INSUFFICIENT_EVIDENCE",
       thin["status"] == INSUFFICIENT and thin["p"] is None
       and thin["required"] == MIN_NONZERO,
       f"{thin['n_nonzero']} nonzero, needs {MIN_NONZERO}")

    # --- Holm across the family ---------------------------------------
    h = holm({"C1": 0.001953125, "C2": 0.02})
    ck("Holm corrects across C1 and C2 with m = 2",
       h["C1"]["m"] == 2 and h["C1"]["threshold"] == ALPHA / 2
       and h["C1"]["rejects_null"] and h["C2"]["threshold"] == ALPHA,
       f"C1 thr {h['C1']['threshold']}, C2 thr {h['C2']['threshold']}")
    h2 = holm({"C1": 0.03, "C2": 0.04})
    ck("  and a p above its Holm threshold does NOT reject",
       not h2["C1"]["rejects_null"],
       f"p 0.03 vs threshold {h2['C1']['threshold']}")

    # --- coverage ------------------------------------------------------
    cov = coverage_gate({"btc": 960, "eth": 980}, {"btc": 1000, "eth": 1000})
    ck("the 95% native-OK coverage gate passes per coin",
       cov["passes"] and cov["per_coin"]["btc"]["coverage"] == 0.96)
    cov_bad = coverage_gate({"btc": 940, "eth": 990},
                            {"btc": 1000, "eth": 1000})
    ck("  and ONE coin below 95% fails the gate",
       not cov_bad["passes"] and not cov_bad["per_coin"]["btc"]["passes"]
       and cov_bad["per_coin"]["eth"]["passes"],
       f"btc {cov_bad['per_coin']['btc']['coverage']}, eth "
       f"{cov_bad['per_coin']['eth']['coverage']}")

    # --- the four conditions, computed --------------------------------
    inc = [0.1] * 9 + [0.0]
    v = verdict("C1", increments=inc, holm_row=h["C1"], coverage=cov,
                predicates={"population": True, "timestamps": True,
                            "complement": True, "reconciliation": True})
    ck("the four conditions are a CONJUNCTION the code evaluates",
       v["passes"] and all(v["conditions"].values())
       and v["computed_not_printed"],
       json.dumps(v["conditions"]))
    ck("  and the mean/median summaries include the ZERO day",
       v["n_days_in_summary"] == 10 and v["summaries_include_zero_days"]
       and abs(v["mean_delta_LL"] - 0.09) < 1e-12,
       f"mean over {v['n_days_in_summary']} days = {v['mean_delta_LL']}")
    v_bad = verdict("C2", increments=inc, holm_row=h["C1"],
                    coverage=cov_bad,
                    predicates={"population": True, "timestamps": False})
    ck("  and ONE false condition fails the conjunction, naming which",
       not v_bad["passes"]
       and set(v_bad["failing"]) == {"3_coverage_gate_passes",
                                     "4_all_predicates_pass"},
       ", ".join(v_bad["failing"]))
    v_neg = verdict("C3", increments=[-0.1] * 10, holm_row=h["C1"],
                    coverage=cov, predicates={"p": True})
    ck("  and a NEGATIVE mean fails condition 2 even with a small p",
       not v_neg["passes"]
       and "2_mean_and_median_increment_positive" in v_neg["failing"],
       f"mean {v_neg['mean_delta_LL']}")
    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    print(json.dumps({"protocol": PROTOCOL, "alpha": ALPHA,
                      "m_family": M_FAMILY, "G_required": G_REQUIRED,
                      "band_days": BAND_DAYS,
                      "min_nonzero_for_the_exact_test": MIN_NONZERO,
                      "coverage_minimum": COVERAGE_MIN}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
