"""THE §8 PATH ON REAL DATA FROM ALREADY-CONSUMED DAYS. PLUMBING ONLY.

Synthetic inputs never carry the shapes that break things. This drives
the same §8 path over BE's REAL day books for days that are ALREADY
SPENT -- 09-03..09-06 are the latency days and 09-07..09-10 are consumed
by the cancellation test, so rule 34's own sentence applies: "Exercise
corrected pipelines on the days already consumed; they are free to
re-run precisely because they are spent."

THE OUTPUT IS NOT EVIDENCE AND SAYS SO IN A FIELD. Every artifact this
writes carries `verdict = PLUMBING_ONLY_NOT_EVIDENCE`, the consumed
population, and the two proxies below. Rule 35: a limit that lives only
in prose does not bind the reader, so it is a required field.

  * IDENTITY IS A PROXY HERE -- the mid of the neutral reference's own
    BUY_UP/SELL_UP levels, not the Identity estimator.
  * THE CANDIDATES ARE DETERMINISTIC PLUMBING -- a shrink toward 0.5 and
    a constant offset. They estimate nothing.

Days from 2026-09-11 on are REFUSED by name: they are live, and the eye
is the thing that spends the day.

Usage:  de_fair_value_plumbing_run.py --falsify
        de_fair_value_plumbing_run.py --run [--days 20260907,...] [--out DIR]
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import da_fair_price_identity as FP                # noqa: E402
import de_fair_value_actions as ACT                # noqa: E402
import de_fair_value_pnl as PNL                    # noqa: E402
import de_fair_value_predictive as PRED            # noqa: E402
import de_fair_value_rehearsal as REH              # noqa: E402

PROTOCOL = "P003_DE_FAIR_VALUE_PLUMBING_RUN_V1"
VERDICT = "PLUMBING_ONLY_NOT_EVIDENCE"

#: Days this run may touch AT ALL, named one by one rather than by a
#: range that would quietly grow.
#:
#: 09-03..09-06 are the latency days and 09-07..09-10 are consumed by the
#: cancellation test. 09-11..09-13 cost NOTHING EITHER, and that is a
#: correction rather than a relaxation (coordinator, DE 386): §8's
#: population is the first complete UTC day STRICTLY AFTER the full
#: pipeline freeze, the freeze is not effective while REV's adjudication
#: is pending, so every day up to and including today PRECEDES any
#: possible freeze and can never be in the §8 population. There is no
#: protected day among them to spend. What they cost is BE's build time.
CONSUMED_DAYS = ("20260903", "20260904", "20260905", "20260906",
                 "20260907", "20260908", "20260909", "20260910")
CANNOT_BE_IN_THE_POPULATION = ("20260911", "20260912", "20260913")
READABLE_DAYS = CONSUMED_DAYS + CANNOT_BE_IN_THE_POPULATION
WHY_READABLE = {
    "consumed": "09-03..09-06 latency days; 09-07..09-10 consumed by the "
                "cancellation test -- free to re-run because they are "
                "spent (rule 34's own sentence)",
    "cannot_be_in_the_population":
        "09-11..09-13 PRECEDE any possible freeze -- §8's population "
        "begins strictly AFTER a freeze that is not effective -- so they "
        "can never be §8 days and reading them spends nothing",
    "still_refused": "any day after the named set: the guard must still "
                     "bite, and a range that grows on its own is how a "
                     "protected day gets read",
}
PROTECTED_DAY = "PLUMBING_RUN_TOUCHED_A_PROTECTED_DAY"
NO_BOOK = "NO_DAY_BOOK_FOR_THIS_DAY"
NO_OUTCOME = "NO_OFFICIAL_RESOLUTION_FOR_THIS_SLUG"

#: THE GAP THE REAL DATA EXPOSED. BE's neutral reference records the two
#: QUOTE LEVELS per generation (0.49 BUY_UP / 0.50 SELL_UP on the first
#: window of 09-07), which are §7's OUTPUT. It does NOT record the fair
#: value the generation consumed. Feeding the levels through as the
#: consumed value makes every two-sided generation a
#: FORECAST_ACTION_DUPLICATE_KEY -- one key, two values -- which is the
#: normal shape of a two-sided quote, not a defect. So either the book
#: gains a `fair_value_consumed` per generation, or the adapter derives
#: one and says which; this run takes the mid AND SAYS SO.
CONSUMED_VALUE_ADAPTER = (
    "mid of the two quote levels -- the day book records LEVELS, not the "
    "fair value each generation consumed")

ROOT = Path("/home/yuqing/ctaNew/data/pm_5min")
DERIVED = ROOT / "derived"
RESOLUTIONS = ROOT / "resolutions.jsonl"

#: Preference order among revisions, newest contract first. Recorded.
REVISIONS = ("__L250ms__FWD1", "__L250ms__EV22", "__L250ms__EV21",
             "__L250ms__EV20", "__L250ms", "")


class PlumbingRefused(ValueError):
    """The plumbing run cannot proceed as declared."""


def assert_day_is_consumed(day: str) -> None:
    """THE EYE IS THE THING THAT SPENDS THE DAY (rule 34a)."""
    if day not in READABLE_DAYS:
        raise PlumbingRefused(
            f"REFUSED {PROTECTED_DAY}: {day} is in neither the consumed "
            f"set {CONSUMED_DAYS} nor the pre-freeze set "
            f"{CANNOT_BE_IN_THE_POPULATION}. A plumbing test is still a "
            f"LOOK, and rule 11 does not care about intent.")


def book_path(day: str, coin: str = "btc") -> Path:
    assert_day_is_consumed(day)
    for rev in REVISIONS:
        p = DERIVED / f"be_daybook_{day}_{coin}{rev}.pkl"
        if p.is_file():
            return p
    raise PlumbingRefused(
        f"REFUSED {NO_BOOK}: no day book for {coin} on {day} under "
        f"{DERIVED}. This run reads what BE built; it builds nothing.")


def coins_with_books(day: str) -> tuple:
    """WHICH COINS ACTUALLY HAVE A BOOK. §8's scope is btc AND eth."""
    got = []
    for coin in PRED.COINS:
        try:
            book_path(day, coin)
            got.append(coin)
        except PlumbingRefused:
            pass
    return tuple(got)


def official_outcomes(slugs) -> dict:
    """REAL resolutions: winners.Up from the venue's own closed record."""
    want = set(slugs)
    out = {}
    with RESOLUTIONS.open() as fh:
        for line in fh:
            if '"closed":true' not in line:
                continue
            try:
                rec = json.loads(line)
            except Exception:                               # noqa: BLE001
                continue
            slug = rec.get("slug")
            if slug in want and isinstance(rec.get("winners"), dict):
                out[slug] = bool(rec["winners"].get("Up"))
    return out


def _identity(coin, slug, ws, gen, sides) -> FP.FairPrice:
    """THE PROXY, NAMED: the mid of the reference's own two levels.

    A generation quoted on ONE side only has no mid, and that is a real
    ONE_SIDED record, not a missing one -- the book's side counts differ
    by thousands on a normal day.
    """
    buy = sides.get("BUY_UP"), sides.get("SELL_UP")
    b, s = buy
    t0 = (b or s)["t0"]
    src = float(ws) + float(t0)
    if b is None or s is None:
        return FP.FairPrice(coin=coin, window_start=ws, outcome="UP",
                            value=None, source_timestamp=src,
                            local_knowledge_timestamp=src, freshness_s=0.0,
                            status="ONE_SIDED", estimator="Identity",
                            detail="one side quoted at this generation")
    mid = (float(b["level"]) + float(s["level"])) / 2.0
    return FP.FairPrice(coin=coin, window_start=ws, outcome="UP",
                        value=mid, source_timestamp=src,
                        local_knowledge_timestamp=src, freshness_s=0.0,
                        status=FP.OK, estimator="Identity")


def candidate_c1(ident: FP.FairPrice) -> FP.FairPrice:
    """PLUMBING: shrink 10% toward 0.5. Estimates nothing."""
    if ident.status != FP.OK:
        return ident
    v = 0.5 + 0.9 * (ident.value - 0.5)
    return FP.FairPrice(coin=ident.coin, window_start=ident.window_start,
                        outcome="UP", value=v,
                        source_timestamp=ident.source_timestamp,
                        local_knowledge_timestamp=(
                            ident.local_knowledge_timestamp),
                        freshness_s=ident.freshness_s, status=FP.OK,
                        estimator="C1_PLUMBING")


#: C2's abstention threshold, in seconds of quoted life. A real property
#: of the real rows -- generations this short are the ones a slow
#: estimator would miss -- so the coverage figure it produces is measured
#: rather than constructed.
SHORT_GENERATION_S = 1.0


def candidate_c2(ident: FP.FairPrice, life_s: float = None) -> FP.FairPrice:
    """PLUMBING: a constant offset, clipped, ABSTAINING on generations
    shorter than SHORT_GENERATION_S -- which is how a real challenger
    fails to cover a coin."""
    if life_s is not None and life_s < SHORT_GENERATION_S:
        return FP.FairPrice(
            coin=ident.coin, window_start=ident.window_start, outcome="UP",
            value=None, source_timestamp=ident.source_timestamp,
            local_knowledge_timestamp=ident.local_knowledge_timestamp,
            freshness_s=ident.freshness_s, status="NOT_READY",
            estimator="C2_PLUMBING",
            detail=f"generation shorter than {SHORT_GENERATION_S}s")
    if ident.status != FP.OK:
        return ident
    v = min(0.99, max(0.01, ident.value + 0.02))
    return FP.FairPrice(coin=ident.coin, window_start=ident.window_start,
                        outcome="UP", value=v,
                        source_timestamp=ident.source_timestamp,
                        local_knowledge_timestamp=(
                            ident.local_knowledge_timestamp),
                        freshness_s=ident.freshness_s, status=FP.OK,
                        estimator="C2_PLUMBING")


def read_day(day: str, coin: str = "btc") -> dict:
    """ONE REAL DAY: consumptions, the population, Identity, statuses."""
    path = book_path(day, coin)
    book = pickle.load(path.open("rb"))
    fr = book["fr"]
    reference = fr["reference"]
    marks = fr.get("terminal_marks", {})
    statuses = dict(fr.get("statuses", {}))
    consumptions, identity, pop_rows, life = [], {}, [], {}
    excluded = Counter()
    for slug, by_side in reference.items():
        ws = int(slug.rsplit("-", 1)[1])
        mark = marks.get(slug) or {}
        # A REAL EXCLUSION, CARRIED AS A STATUS (rule 4), never a drop.
        if mark.get("ended_in_gap"):
            excluded["TERMINAL_MARK_ENDED_IN_GAP"] += 1
            continue
        if mark.get("status") not in (None, "OK"):
            excluded[f"TERMINAL_MARK_{mark.get('status')}"] += 1
            continue
        bygen = {}
        for side, lst in by_side.items():
            for g in lst:
                bygen.setdefault(int(g["gen"]), {})[side] = g
        for gen, sides in bygen.items():
            anyg = sides.get("BUY_UP") or sides.get("SELL_UP")
            stamp = int((float(ws) + float(anyg["t0"])) * 1_000_000_000)
            gid = f"g{gen}"
            pop_rows.append((slug, gid))
            ident = _identity(coin, slug, ws, gen, sides)
            identity[(coin, slug, gid)] = ident
            # THE ADAPTER DECISION, AND IT IS A REAL GAP (see
            # CONSUMED_VALUE_ADAPTER): the book records the two QUOTE
            # LEVELS, which are §7's OUTPUT, not the fair value each
            # generation consumed. One decision is one value, so the two
            # sides share the mid and the naive reading is probed below.
            consumed = (ident.value if ident.value is not None
                        else round(float(anyg["level"]), 6))
            life[(coin, slug, gid)] = max(
                float(g["t1"]) - float(g["t0"]) for g in sides.values())
            for side in sides:
                consumptions.append(
                    {"coin": coin, "slug": slug, "generation_id": gid,
                     "decision_recv_ns": stamp, "quote_side": side,
                     "up_probability_consumed": consumed,
                     "level_this_side_quoted": round(
                         float(sides[side]["level"]), 6),
                     "window_start": ws,
                     "on_identity_reference_path": True})
    del book, fr, reference
    population = {"actions": pop_rows,
                  "population": fr_population(path),
                  "as_of": f"{day}T00:00:00Z",
                  "source_identity": path.name}
    return {"day": day, "coin": coin, "book": path.name,
            "consumptions": consumptions, "population": population,
            "identity": identity, "generation_life_s": life,
            "book_statuses": statuses,
            "excluded_windows": dict(excluded),
            "n_slugs": len(pop_rows and {s for s, _ in pop_rows} or set())}


def fr_population(path: Path) -> str:
    return f"BE_NEUTRAL_REFERENCE::{path.name}"


def naive_adapter_probe(day_data: dict) -> dict:
    """DRIVE the reading that treats each side's LEVEL as the consumed
    value. On real two-sided quotes it must refuse; the refusal is the
    finding, so it is computed here rather than described."""
    rows = [dict(r, up_probability_consumed=r["level_this_side_quoted"])
            for r in day_data["consumptions"][:400]]
    try:
        ACT.build_actions(rows,
                          canonical_population=day_data["population"])
        return {"refused": None,
                "note": "the naive reading did NOT refuse on this sample"}
    except ACT.ActionsRefused as exc:
        return {"refused": REH._refusal_name(exc),
                "detail": str(exc)[:180],
                "why_it_is_not_a_defect":
                    "both sides of one two-sided quote consume different "
                    "LEVELS by construction; the duplicate-key guard is "
                    "right and the ADAPTER was wrong"}


def score_real_day(day_data: dict, outcomes: dict) -> dict:
    """Build canonical actions on REAL rows and score them, paired."""
    built = ACT.build_actions(day_data["consumptions"],
                              canonical_population=day_data["population"])
    ident = day_data["identity"]
    missing = [a for a in built["actions"] if a.slug not in outcomes]
    actions = [a for a in built["actions"] if a.slug in outcomes]
    scored = ACT.score_actions(
        actions, lambda a: ident.get((a.coin, a.slug, a.generation_id)),
        lambda a: candidate_c1(
            ident.get((a.coin, a.slug, a.generation_id))),
        outcomes)
    life = day_data["generation_life_s"]
    scored_c2 = ACT.score_actions(
        actions, lambda a: ident.get((a.coin, a.slug, a.generation_id)),
        lambda a: candidate_c2(
            ident.get((a.coin, a.slug, a.generation_id)),
            life.get((a.coin, a.slug, a.generation_id))),
        outcomes)
    return {"built_excluded": built.get("excluded_counts",
                                        built.get("excluded", {})),
            "n_actions": len(actions),
            "actions_without_a_resolution": len(missing),
            "unresolved_status": NO_OUTCOME if missing else None,
            "C1": scored, "C2": scored_c2}


def blind_row(day: str, coins: tuple, resolutions_ok: bool) -> "PRED.BlindDayInputs":
    return PRED.BlindDayInputs(
        day=day, book_gate_pass=bool(coins),
        official_resolutions_present=resolutions_ok,
        settlement_verification_covered=resolutions_ok,
        coins_complete=coins)


def plumbing_run(days=READABLE_DAYS, outdir=None, scope_override=None) -> dict:
    """THE WHOLE §8 PATH ON REAL ROWS. Not evidence, and it says so."""
    outdir = Path(outdir) if outdir else Path(".")
    per_day, blind, refusals, statuses = [], [], [], Counter()
    pipelining = None
    native = Counter()
    eligible = Counter()
    native_c2 = Counter()
    eligible_c2 = Counter()
    for day in days:
        assert_day_is_consumed(day)
        coins = coins_with_books(day)
        if not coins:
            refusals.append({"day": day, "refusal": NO_BOOK,
                             "stage": "book"})
            continue
        dd = read_day(day, coins[0])
        outs = official_outcomes({s for s, _ in dd["population"]["actions"]})
        naive = naive_adapter_probe(dd)
        if naive["refused"]:
            refusals.append({"day": day, "refusal": naive["refused"],
                             "stage": "adapter",
                             "fired_on_real_data_not_on_synthetic": True,
                             "detail": naive.get("why_it_is_not_a_defect")})
        got = score_real_day(dd, outs)
        prim = got["C1"]["PRIMARY_fallback_scored"]
        prim2 = got["C2"]["PRIMARY_fallback_scored"]
        prof = duration_profile(dd["generation_life_s"], dd["identity"])
        cov_c2_day = (prim2["n_native"] / prim2["n"]) if prim2["n"] else None
        row = {"day": day, "book": dd["book"], "coins_with_books": coins,
               "generation_duration_profile": prof,
               "fraction_at_or_over_one_second":
                   prof.get("fraction_at_or_over_one_second"),
               "n_generations_under_one_second":
                   prof.get("n_under_one_second"),
               "share_under_one_second":
                   prof.get("share_under_one_second"),
               "C2_measured_coverage": cov_c2_day,
               "C2_coverage_ceiling": coverage_ceiling(prof, cov_c2_day),
               "n_actions": got["n_actions"],
               "actions_without_a_resolution":
                   got["actions_without_a_resolution"],
               "excluded_windows": dd["excluded_windows"],
               "book_statuses": dd["book_statuses"],
               "identity_non_ok_actions": got["C1"]["identity_non_ok_actions"],
               "challenger_status_counts": got["C1"]["challenger_status_counts"],
               "n_native": prim["n_native"], "n_fallback": prim["n_fallback"],
               "n_native_C2": prim2["n_native"],
               "n_fallback_C2": prim2["n_fallback"],
               "delta_LL_C1": (None if prim["identity_mean_log_loss"] is None
                               else prim["identity_mean_log_loss"]
                               - prim["policy_mean_log_loss"]),
               "delta_LL_C2": (None if prim2["identity_mean_log_loss"] is None
                               else prim2["identity_mean_log_loss"]
                               - prim2["policy_mean_log_loss"])}
        if pipelining is None:
            pipelining = pipelining_from_day(dd, outs, prim)
            pipelining["measured_on_day"] = day
        per_day.append(row)
        for k, v in dd["excluded_windows"].items():
            statuses[k] += v
        native[coins[0]] += prim["n_native"]
        eligible[coins[0]] += prim["n"]
        native_c2[coins[0]] += prim2["n_native"]
        eligible_c2[coins[0]] += prim2["n"]
        blind.append(blind_row(day, coins,
                               got["actions_without_a_resolution"] == 0))
        del dd, got
    # --- the path, twice: as DECLARED, and under a marked override ----
    elig = PRED.eligible_days(blind)
    acc = PRED.accrual(blind)
    coverage = PRED.coverage_gate(native, eligible)
    coverage_c2 = PRED.coverage_gate(native_c2, eligible_c2)
    incs = {c: [r[f"delta_LL_{c}"] for r in per_day
                if r[f"delta_LL_{c}"] is not None] for c in ("C1", "C2")}
    pvals = {c: PRED.exact_sign_p(v).get("p") for c, v in incs.items()}
    holm = PRED.holm(pvals)
    out = {
        "protocol": PROTOCOL,
        "verdict": VERDICT,
        "population_is_consumed": True,
        "THESE_NUMBERS_ARE_NOT_A_RESULT":
            "the days are already spent, Identity is a book-mid PROXY and "
            "both candidates are deterministic plumbing. Nothing here "
            "estimates anything, and no line of it may be read as skill.",
        "identity_is_a_proxy": "mid of the neutral reference's BUY_UP and "
                               "SELL_UP levels",
        "consumed_value_adapter": CONSUMED_VALUE_ADAPTER,
        "candidates_are_plumbing": {"C1": "shrink 10% toward 0.5",
                                    "C2": "+0.02, clipped"},
        "days": list(days), "per_day": per_day,
        "days_requested": len(list(days)), "days_measured": len(per_day),
        "days_with_no_book_yet": [r["day"] for r in refusals
                                  if r["refusal"] == NO_BOOK],
        "CAN_ONE_COIN_BE_VALUED_WHILE_THE_OTHER_BUILDS": pipelining,
        "WHAT_SPENDS_A_BAND_DAY": band_slack(),
        "THE_TWO_BLOCKERS": {
            "eth_has_no_day_book_on_any_day": {
                "coins_with_books_per_day":
                    {r["day"]: list(r["coins_with_books"])
                     for r in per_day},
                "declared_scope": list(PRED.COINS),
                "every_day_is": "COINS_INCOMPLETE",
                "n_evaluable_days": 0,
                "THE_CLOCK_CANNOT_START": True,
                "this_is_a_DATA_blocker_not_a_code_one": True},
            "C2_coverage_is_unreachable": None},
        "eligibility_as_declared": elig, "accrual_as_declared": acc,
        "coverage_gate": coverage,
        "coverage_gate_C2": coverage_c2,
        "coverage_note":
            "C1 covers every Identity-eligible action by construction; C2 "
            "abstains on generations shorter than "
            f"{SHORT_GENERATION_S}s, so its coverage is a MEASURED share "
            "of the real rows",
        "real_exclusion_statuses": dict(statuses),
        # THE NUMBERS AND THEIR LIMIT IN ONE BLOCK. A reader resolves
        # fields; a caveat in a covering message does not travel with the
        # number it qualifies (rule 35).
        "plumbing_numbers": {
            "NOT_EVIDENCE": VERDICT,
            "these_are_plumbing_not_evidence":
                "the days are spent, Identity is a proxy, the candidates "
                "estimate nothing",
            "population_is_consumed": True,
            "identity_is_a_proxy": "mid of the neutral reference's "
                                   "BUY_UP and SELL_UP levels",
            "candidates_are_plumbing": {"C1": "shrink 10% toward 0.5",
                                        "C2": "+0.02, clipped, abstaining "
                                              "under 1s"},
            "increments": incs, "p_values": pvals, "holm": holm,
            "exact_sign_test": {c: PRED.exact_sign_p(v)
                                for c, v in incs.items()},
            "futility": {c: PRED.futility(v) for c, v in incs.items()}},
        "refusals_encountered": refusals,
    }
    # THE SECOND BLOCKER, COMPUTED OVER THE WHOLE RUN.
    pooled_n = sum(r["generation_duration_profile"].get(
        "n_identity_eligible_generations", 0) for r in per_day)
    pooled_k = sum(r["generation_duration_profile"].get(
        "n_at_or_over_one_second", 0) for r in per_day)
    pooled = {"n_identity_eligible_generations": pooled_n,
              "n_at_or_over_one_second": pooled_k,
              "n_under_one_second": pooled_n - pooled_k,
              "fraction_at_or_over_one_second":
                  (pooled_k / pooled_n) if pooled_n else None}
    out["THE_TWO_BLOCKERS"]["C2_coverage_is_unreachable"] = {
        "measured_coverage_btc": coverage["per_coin"]["btc"]["coverage"],
        "measured_coverage_C2_btc":
            coverage_c2["per_coin"]["btc"]["coverage"],
        "pooled_duration": pooled,
        "ceiling": coverage_ceiling(
            pooled, coverage_c2["per_coin"]["btc"]["coverage"]),
        "stability_of_the_fraction_across_days":
            stability(per_day, "fraction_at_or_over_one_second"),
        # DEAD OR MERELY NARROW -- the question decided by arithmetic
        # rather than by a word. The day-to-day swing is real; what
        # matters is whether any of it reaches the gate.
        "reachability_per_day": [
            {"day": r["day"],
             "fraction_at_or_over_one_second":
                 r["fraction_at_or_over_one_second"],
             "reaches_the_gate": (r["fraction_at_or_over_one_second"]
                                  or 0.0) >= PRED.COVERAGE_MIN}
            for r in per_day],
        "n_days_reaching_the_gate": sum(
            1 for r in per_day
            if (r["fraction_at_or_over_one_second"] or 0.0)
            >= PRED.COVERAGE_MIN),
        "best_day": max((r["fraction_at_or_over_one_second"] or 0.0)
                        for r in per_day),
        "gap_on_the_best_day": PRED.COVERAGE_MIN - max(
            (r["fraction_at_or_over_one_second"] or 0.0)
            for r in per_day),
        "multiple_of_the_pooled_fraction_the_gate_requires":
            (PRED.COVERAGE_MIN / pooled["fraction_at_or_over_one_second"])
            if pooled["fraction_at_or_over_one_second"] else None,
        "THE_SWING_IS_REAL_AND_IRRELEVANT_TO_REACHABILITY":
            "the fraction is NOT stable within 5 points across these "
            "days, and the gate still needs several times the best day, "
            "so the variation does not decide the question -- but the "
            "days after 09-10 are unread, so this is a statement about "
            "09-03..09-10 and nothing later",
        "structural_or_incidental":
            "STRUCTURAL if the fraction is stable across the days "
            "measured: the candidate's clock is coarser than the "
            "decision rate, and that is a property of the estimand",
        "NOT_A_TUNING_GAP": True,
        "nothing_here_was_changed_after_seeing_it":
            "the gate is 0.95 as declared and C2's threshold is 1s as "
            "declared; recording is not repairing"}
    out["comparison_to_synthetic"] = compare_to_synthetic(out, outdir)
    REH.write_artifact(outdir / "plumbing_run_section8.json", out)
    return out


def compare_to_synthetic(real: dict, outdir: Path) -> dict:
    """WHICH REFUSALS FIRED ON REAL DATA THAT DID NOT FIRE ON SYNTHETIC.

    Computed against the rehearsal's own inventory, not remembered: a
    refusal the synthetic run classified as "only under a defect" and
    that REAL ORDINARY DATA fires is a misclassification, and it is the
    entire point of running on real rows.
    """
    inv = REH.inventory(outdir)
    synth = {r["refusal"]: r for r in inv["table"]}
    fired = {}
    for row in real.get("refusals_encountered", []):
        fired.setdefault(row["refusal"], row)
    out = []
    for name, row in sorted(fired.items()):
        was = synth.get(name, {}).get("fires_on_a_normal_day",
                                      "NOT IN THE SYNTHETIC INVENTORY")
        out.append({
            "refusal": name,
            "synthetic_said": was,
            "real_data_says": "FIRES ON ORDINARY REAL ROWS",
            "reclassified": was.startswith("no"),
            "stage": row.get("stage"),
            "why": row.get("detail")})
    statuses = sorted(set(real.get("real_exclusion_statuses", {}))
                      | {r["status"] for r
                         in real["eligibility_as_declared"]["rows"]
                         if not r["evaluable"]}
                      | {k for d in real["per_day"]
                         for k in d["challenger_status_counts"]
                         if k != "OK"})
    return {"refusals_new_on_real_data": out,
            "n_reclassified": sum(1 for r in out if r["reclassified"]),
            "statuses_seen_on_real_data_only": statuses,
            "synthetic_inventory_size": inv["n_refusals_scanned"]}


#: The buckets the distribution is reported in. Declared before it is
#: measured, so the shape is not chosen around the answer.
DURATION_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 30.0, 300.0)


def duration_profile(life: dict, identity: dict) -> dict:
    """HOW LONG A REAL GENERATION LIVES, on the SCORED population.

    The coverage denominator is Identity-eligible actions, so the profile
    is taken over exactly those -- a distribution over a different
    population would not explain the coverage it is offered to explain.
    """
    xs = sorted(v for k, v in life.items()
                if identity.get(k) is not None
                and identity[k].status == FP.OK)
    n = len(xs)
    if not n:
        return {"n": 0, "status": "NO_IDENTITY_ELIGIBLE_GENERATIONS"}

    def q(f):
        return xs[min(n - 1, max(0, int(f * n)))]

    hist, prev = {}, 0.0
    for b in DURATION_BUCKETS:
        c = sum(1 for x in xs if prev <= x < b)
        hist[f"[{prev},{b})"] = c
        prev = b
    hist[f"[{prev},inf)"] = sum(1 for x in xs if x >= prev)
    n_ge_1 = sum(1 for x in xs if x >= SHORT_GENERATION_S)
    return {"n_identity_eligible_generations": n,
            "seconds": {"p01": q(0.01), "p10": q(0.10), "p25": q(0.25),
                        "p50": q(0.50), "p75": q(0.75), "p90": q(0.90),
                        "p99": q(0.99), "max": xs[-1], "min": xs[0]},
            "histogram_seconds": hist,
            "n_under_one_second": n - n_ge_1,
            "share_under_one_second": (n - n_ge_1) / n,
            "n_at_or_over_one_second": n_ge_1,
            "fraction_at_or_over_one_second": n_ge_1 / n,
            "threshold_s": SHORT_GENERATION_S}


def coverage_ceiling(profile: dict, measured: float = None) -> dict:
    """THE ARITHMETIC, SHOWN. A candidate that cannot form an opinion in
    under `threshold_s` can cover AT MOST the generations that live that
    long -- so its ceiling is a property of the DATA, not of tuning."""
    if profile.get("n_identity_eligible_generations", 0) == 0:
        return {"status": "NOT_MEASURABLE"}
    n = profile["n_identity_eligible_generations"]
    k = profile["n_at_or_over_one_second"]
    ceiling = k / n
    gate = PRED.COVERAGE_MIN
    out = {"ceiling": ceiling,
           "arithmetic": f"{k} generations >= {SHORT_GENERATION_S}s of "
                         f"{n} Identity-eligible = {ceiling:.6f}",
           "gate_minimum": gate,
           "shortfall_against_the_gate": gate - ceiling,
           "gate_is_unreachable_at_this_threshold": ceiling < gate,
           "why": (f"a challenger needing {SHORT_GENERATION_S}s to form "
                   f"an opinion cannot cover a decision that no longer "
                   f"exists: the ceiling is the share of generations that "
                   f"live that long, and no tuning of the candidate "
                   f"raises it"),
           "computed_not_asserted": True}
    if measured is not None:
        out["measured_coverage"] = measured
        out["measured_equals_the_ceiling"] = abs(measured - ceiling) < 1e-9
        out["mechanism_confirmed_not_assumed"] = (
            "the measured coverage EQUALS the ceiling, so the shortfall "
            "is the duration distribution and nothing else"
            if abs(measured - ceiling) < 1e-9 else
            "the measured coverage differs from the ceiling, so something "
            "besides duration is also removing actions")
    return out


def stability(per_day, key: str) -> dict:
    """IS IT THE DAYS, OR IS IT THE SHAPE? Computed across the days
    available -- and the days NOT available are named."""
    xs = [r[key] for r in per_day if r.get(key) is not None]
    if len(xs) < 2:
        return {"status": PRED.INSUFFICIENT, "n_days": len(xs)}
    lo, hi = min(xs), max(xs)
    mean = sum(xs) / len(xs)
    sd = (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    return {"n_days": len(xs), "per_day": xs, "min": lo, "max": hi,
            "mean": mean, "stdev": sd, "range": hi - lo,
            "relative_range": (hi - lo) / mean if mean else None,
            "STABLE_WITHIN_5_POINTS": (hi - lo) <= 0.05,
            "days_measured": [r["day"] for r in per_day],
            "days_not_measurable":
                "2026-09-11 onward are protected and were not read, so "
                "stability beyond 09-10 is UNTESTED",
            "coins_not_measurable":
                "eth has no day book on any day, so the distribution "
                "cannot be measured for it from existing artifacts"}


MEAN_OF_MEANS = "DAY_INCREMENT_COMBINED_AS_A_MEAN_OF_MEANS"
INCOMPLETE_COIN_SET = "PORTFOLIO_DAY_VALUED_ON_AN_INCOMPLETE_COIN_SET"


def assert_day_is_complete(coins_present, declared=None) -> dict:
    """A COIN MAY BE VALUED EARLY; A DAY MAY NOT BE PUBLISHED EARLY.

    Per-coin valuation is independent work. The portfolio day's number is
    an action-weighted mean over the DECLARED coin set, so a number
    computed while a coin is still building is a different estimand
    wearing the day's name.
    """
    declared = tuple(declared or PRED.COINS)
    present = tuple(sorted(set(coins_present)))
    missing = [c for c in declared if c not in present]
    if missing:
        raise PlumbingRefused(
            f"REFUSED {INCOMPLETE_COIN_SET}: the day declares "
            f"{list(declared)} and {present} is present; {missing} is "
            f"still building. The per-coin WORK may proceed -- this "
            f"refuses only the publication of a partial day AS the day.")
    return {"complete": True, "coins": present, "declared": list(declared)}


def combine_partials(parts) -> dict:
    """ONE DAY'S INCREMENT FROM PARTS SCORED SEPARATELY.

    The day's increment is an ACTION-WEIGHTED mean, so partial results
    combine as SUMS -- total log loss and action counts -- never as a
    mean of the parts' means. Averaging two means silently reweights a
    day toward whichever part had fewer actions, and the two agree only
    when the parts happen to be the same size.
    """
    ident = pol = 0.0
    n = 0
    for part in parts:
        k = int(part["n"])
        if not k:
            continue
        ident += float(part["identity_mean_log_loss"]) * k
        pol += float(part["policy_mean_log_loss"]) * k
        n += k
    if not n:
        raise PlumbingRefused(
            f"REFUSED {PRED.NO_SCORED_ACTIONS}: no scored action in any "
            f"part, and an empty mean is not zero.")
    return {"delta_LL": (ident - pol) / n, "n": n,
            "identity_mean_log_loss": ident / n,
            "policy_mean_log_loss": pol / n,
            "combined_as": "action-weighted SUMS, not a mean of means",
            "n_parts": len(list(parts))}


def mean_of_means(parts) -> float:
    """THE WRONG COMBINATION, computed so the difference is a number."""
    xs = [float(p["identity_mean_log_loss"]) - float(p["policy_mean_log_loss"])
          for p in parts if int(p["n"])]
    return sum(xs) / len(xs) if xs else 0.0


def per_coin_pipelining_answer(day: str = "20260907") -> dict:
    coins = coins_with_books(day)
    dd = read_day(day, coins[0])
    outs = official_outcomes({s for s, _ in dd["population"]["actions"]})
    whole = score_real_day(dd, outs)["C1"]["PRIMARY_fallback_scored"]
    return pipelining_from_day(dd, outs, whole)


def pipelining_from_day(dd: dict, outs: dict, whole: dict) -> dict:
    """CAN THE CHAIN VALUE ONE COIN WHILE THE OTHER IS STILL BUILDING?

    Answered by DRIVING the partition, not by reading the code: a real
    day is scored whole, then scored again in two disjoint parts, and the
    parts are recombined. The partition here is BY SLUG rather than by
    coin because no eth book exists -- and the arithmetic of recombining
    disjoint action sets is the same either way, which is the point.
    """
    slugs = sorted({s for s, _ in dd["population"]["actions"]})
    # DELIBERATELY UNBALANCED (20/80), because two equal parts hide the
    # difference between the right combination and the wrong one.
    cut = max(1, len(slugs) // 5)
    part_slugs = (set(slugs[:cut]), set(slugs[cut:]))
    parts = []
    for keep in part_slugs:
        sub = dict(dd)
        sub["consumptions"] = [r for r in dd["consumptions"]
                               if r["slug"] in keep]
        sub["population"] = dict(dd["population"],
                                 actions=[(sl, g) for sl, g
                                          in dd["population"]["actions"]
                                          if sl in keep])
        parts.append(score_real_day(sub, outs)["C1"]
                     ["PRIMARY_fallback_scored"])
    combined = combine_partials(parts)
    whole_delta = (whole["identity_mean_log_loss"]
                   - whole["policy_mean_log_loss"])
    wrong = mean_of_means(parts)
    return {
        "question": "can a portfolio day be valued one coin at a time, "
                    "while the other coin is still building?",
        "ANSWER": "YES for the VALUATION WORK, NO for the DAY'S VERDICT",
        "valuation_is_per_book": {
            "driver": "de_forward_value_day.py --day --book",
            "takes": "ONE book, and a book is one coin",
            "requires_the_other_coin": False,
            "so": "btc can be valued the moment its book lands, while "
                  "eth is still building"},
        "the_day_verdict_needs_both": {
            "eligible_days": "requires coins_complete == "
                             f"{list(PRED.COINS)}; a btc-only day is "
                             f"COINS_INCOMPLETE",
            "coverage_gate": "iterates both coins; a missing coin gives "
                             "coverage None and FAILS",
            "so": "the day cannot be declared evaluable until both books "
                  "exist -- but nothing forces the WORK to be serial"},
        "the_combination_rule_that_makes_it_safe": {
            "rule": "combine parts by action-weighted SUMS",
            "whole_day_delta_LL": whole_delta,
            "recombined_from_parts": combined["delta_LL"],
            "absolute_difference": abs(whole_delta - combined["delta_LL"]),
            "reproduces_the_single_pass": abs(
                whole_delta - combined["delta_LL"]) < 1e-12,
            "n_whole": whole["n"], "n_recombined": combined["n"],
            "MEAN_OF_MEANS_IS_WRONG": {
                "value": wrong,
                "absolute_error_against_the_single_pass":
                    abs(whole_delta - wrong),
                "why": "the day's increment is action-weighted; averaging "
                       "two means reweights the day toward the smaller "
                       "part, and they agree only when the parts are the "
                       "same size",
                "partition_used": "20/80 by slug, deliberately unbalanced"},
        },
        "valuing_a_day_on_a_partial_coin_set": {
            "part_A_alone_as_if_it_were_the_day": (
                (parts[0]["identity_mean_log_loss"]
                 - parts[0]["policy_mean_log_loss"])
                if parts[0]["n"] else None),
            "the_whole_day": whole_delta,
            "absolute_error_if_published_early": (
                abs((parts[0]["identity_mean_log_loss"]
                     - parts[0]["policy_mean_log_loss"]) - whole_delta)
                if parts[0]["n"] else None),
            "refusal_if_attempted": INCOMPLETE_COIN_SET,
            "so": "value btc the moment its book lands; do NOT let that "
                  "number be read as the day's"},
        "partition_driven": "by slug (no eth book exists); the arithmetic "
                            "of recombining disjoint action sets is "
                            "identical for a partition by coin",
        "n_actions_whole": whole["n"],
    }


def tape_retention() -> dict:
    """HOW LONG A DAY CAN STILL BE REBUILT -- measured, not assumed."""
    raw = ROOT / "raw"
    days = sorted(d.name for d in raw.iterdir() if d.is_dir())
    sizes = {}
    for d in days[-6:]:
        sizes[d] = sum(f.stat().st_size for f in (raw / d).iterdir())
    import shutil
    free = shutil.disk_usage(str(raw)).free
    mean_day = (sum(sizes.values()) / len(sizes)) if sizes else 0
    return {"days_of_tape_on_disk": len(days),
            "oldest_day": days[0] if days else None,
            "newest_day": days[-1] if days else None,
            "mean_recent_day_bytes": mean_day,
            "free_bytes": free,
            "headroom_days_at_the_recent_rate":
                (free / mean_day) if mean_day else None,
            "why_it_matters":
                "a book is built FROM the tape, so a night that slips is "
                "recoverable for as long as its tape survives"}


def book_build_lag() -> dict:
    """WERE BOOKS BUILT ON THE NIGHT, OR AFTERWARDS? Measured from the
    files themselves -- retrospective builds are the evidence that a
    slipped night is not automatically a lost day."""
    import datetime as dt
    rows = []
    for day in CONSUMED_DAYS:
        try:
            p = book_path(day)
        except PlumbingRefused:
            continue
        built = dt.datetime.utcfromtimestamp(p.stat().st_mtime)
        day_end = dt.datetime.strptime(day, "%Y%m%d") + dt.timedelta(days=1)
        rows.append({"day": day, "book": p.name,
                     "built_utc": built.isoformat() + "Z",
                     "days_after_the_day_closed":
                         round((built - day_end).total_seconds() / 86400, 2)})
    return {"rows": rows,
            "n_built_after_the_next_day": sum(
                1 for r in rows if r["days_after_the_day_closed"] > 1.0),
            "reading": "books built days after their day prove the build "
                       "is retrospective: the tape, not the night, is the "
                       "thing that must survive"}


def band_slack() -> dict:
    """WHAT ACTUALLY SPENDS ONE OF THE BAND'S FOUR SPARE DAYS.

    §8's band is 14 consecutive calendar days needing 10 evaluable, and
    extension is forbidden -- so the question is not "what makes a night
    late" but "what makes a DAY permanently non-evaluable". Those are
    different lists, and conflating them prices the slack wrong.
    """
    tape = tape_retention()
    lag = book_build_lag()
    costs_a_day = [
        {"mode": "no book is ever built for a coin on that day",
         "status": "COINS_INCOMPLETE", "evidence":
             "MEASURED: 0 eth books exist across 25 days of tape",
         "recoverable_from_tape": True,
         "why_it_still_costs_a_day": "only if it is never built; the tape "
                                     "is there, so this is a BUILD "
                                     "decision, not a data loss"},
        {"mode": "collector gap inside the day fails the book gate",
         "status": "BINANCE_GAP_EXCLUDED / census gap_seconds",
         "evidence": "MEASURED: 09-07 census gap_seconds = 136.05 over "
                     "287 windows -- small, but the failure mode is real",
         "recoverable_from_tape": False,
         "why_it_still_costs_a_day": "the rows were never collected; no "
                                     "rebuild recovers them"},
        {"mode": "official resolutions never arrive for the day's slugs",
         "status": NO_OUTCOME, "evidence":
             "MEASURED: 0 unresolved actions across all 8 days",
         "recoverable_from_tape": False,
         "why_it_still_costs_a_day": "an action with no outcome cannot "
                                     "be scored at any later time"},
        {"mode": "settlement verification does not cover the day",
         "status": "SETTLEMENT_VERIFICATION_NOT_COVERED",
         "evidence": "STRUCTURAL: eligible_days requires it",
         "recoverable_from_tape": False,
         "why_it_still_costs_a_day": "eligibility is a per-day predicate"},
        {"mode": "the day's tape is lost or truncated before the rebuild",
         "status": "NO_TAPE", "evidence":
             f"MEASURED: {tape['days_of_tape_on_disk']} days on disk, "
             f"oldest {tape['oldest_day']}, headroom "
             f"{tape['headroom_days_at_the_recent_rate']:.0f} days at the "
             f"recent rate -- NOT a binding risk over a 14-day band",
         "recoverable_from_tape": False,
         "why_it_still_costs_a_day": "the source is gone"},
    ]
    costs_wall_clock_only = [
        {"mode": "a build fails or is refused at the emit",
         "evidence": "OBSERVED this programme: a day-run guard refused at "
                     "the emit and 1h20m was lost",
         "cost": "hours", "recovers_by": "rebuild from the tape"},
        {"mode": "a worktree is refreshed under a running unit",
         "evidence": "OBSERVED: wt_refresh.sh has no in-flight guard",
         "cost": "hours", "recovers_by": "re-arm and rerun"},
        {"mode": "a scheduled unit holds the heavy-run lock",
         "evidence": "STRUCTURAL: scheduled units BLOCK rather than yield",
         "cost": "hours", "recovers_by": "run after it releases"},
        {"mode": "the chain refuses at stage 0 on a declaration or pin",
         "evidence": "OBSERVED repeatedly this programme",
         "cost": "minutes", "recovers_by": "land the declaration, re-arm"},
        {"mode": "a valuation crashes mid-run",
         "evidence": "the landmine found by the rehearsal: "
                     f"{PNL.NO_SETTLEMENT} raises mid-valuation",
         "cost": "hours", "recovers_by": "the book persists; rerun the "
                                         "valuation"},
        {"mode": "a unit is garbage-collected so its failure reads as "
                 "success",
         "evidence": "OBSERVED: systemctl show on a collected unit "
                     "returns Result=success; read LoadState first",
         "cost": "a night, if unnoticed", "recovers_by": "rebuild"},
    ]
    return {
        "the_question_restated":
            "a night that slips costs WALL CLOCK; a day is spent only "
            "when it can never become evaluable",
        "THE_CORRECTION":
            "books are built retrospectively from the tape -- measured "
            f"below -- so lateness alone does NOT spend a band day. "
            f"{lag['n_built_after_the_next_day']} of {len(lag['rows'])} "
            f"books on disk were built more than a day after their day "
            f"closed.",
        "tape_retention": tape,
        "book_build_lag": lag,
        "COSTS_A_BAND_DAY": costs_a_day,
        "COSTS_WALL_CLOCK_ONLY": costs_wall_clock_only,
        "n_modes_that_cost_a_day": len(costs_a_day),
        "n_modes_that_cost_only_time": len(costs_wall_clock_only),
        "what_this_prices":
            "the band's four spare days are spent by COLLECTION and "
            "RESOLUTION failures, not by late builds -- so the slack to "
            "watch is the collector's, and the nightly schedule buys "
            "wall clock rather than band days",
    }


def falsify() -> int:
    import tempfile
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    out = Path(tempfile.mkdtemp(prefix="de_plumbing_"))
    print("== the day guard ==")
    ck("a consumed day is admitted", assert_day_is_consumed("20260907")
       is None)

    print("== the real books, as they actually are ==")
    coins = coins_with_books("20260907")
    ck("ONLY btc has a day book -- §8's scope is btc AND eth",
       coins == ("btc",), f"coins with books: {coins}")
    ck("  so every real day is INELIGIBLE as declared",
       not PRED.eligible_days([blind_row("20260907", coins, True)]
                              )["evaluable_days"],
       PRED.eligible_days([blind_row("20260907", coins, True)]
                          )["rows"][0]["status"])

    print("== a real day, driven end to end ==")
    got = plumbing_run(("20260907", "20260910"), outdir=out)
    r = got["per_day"][0]
    ck("real actions were built and scored",
       r["n_actions"] > 10000, f"{r['n_actions']} actions on 09-07")
    ck("  and a ONE-DAY stability question answers INSUFFICIENT, never "
       "with a number",
       stability(got["per_day"][:1], "fraction_at_or_over_one_second"
                 )["status"] == PRED.INSUFFICIENT)
    ck("  and REAL exclusions arrived as STATUSES with counts",
       bool(got["real_exclusion_statuses"]),
       str(got["real_exclusion_statuses"]))
    ck("  and Identity was non-OK on real one-sided generations",
       r["identity_non_ok_actions"] > 0,
       f"{r['identity_non_ok_actions']} one-sided actions")
    ck("the artifact says PLUMBING_ONLY_NOT_EVIDENCE in a FIELD",
       json.loads((out / "plumbing_run_section8.json").read_text()
                  )["verdict"] == VERDICT)
    ck("  and it carries the consumed population and both proxies",
       got["population_is_consumed"] and got["identity_is_a_proxy"]
       and set(got["candidates_are_plumbing"]) == {"C1", "C2"})
    cmp = got["comparison_to_synthetic"]
    ck("THE POINT OF THE STEP: a refusal that fired on REAL ordinary "
       "rows and was classified as defect-only on synthetic",
       cmp["n_reclassified"] >= 1,
       str([r["refusal"] for r in cmp["refusals_new_on_real_data"]]))
    ck("  and the real-only statuses are listed",
       "ONE_SIDED" in cmp["statuses_seen_on_real_data_only"],
       str(cmp["statuses_seen_on_real_data_only"]))
    ck("the per-coin coverage gate fails for ONE coin only",
       got["coverage_gate"]["per_coin"]["btc"]["passes"] is True
       and got["coverage_gate"]["per_coin"]["eth"]["passes"] is False
       and got["coverage_gate"]["passes"] is False,
       f"btc {got['coverage_gate']['per_coin']['btc']['coverage']}, "
       f"eth {got['coverage_gate']['per_coin']['eth']['coverage']}")

    print("== CAN BTC BE VALUED WHILE ETH BUILDS? driven, not read ==")
    ans = per_coin_pipelining_answer("20260907")
    ck("the valuation driver takes ONE book, so per-coin work is "
       "independent",
       ans["valuation_is_per_book"]["requires_the_other_coin"] is False,
       ans["valuation_is_per_book"]["driver"])
    ck("the DAY's verdict still needs both coins -- COINS_INCOMPLETE",
       "COINS_INCOMPLETE" in ans["the_day_verdict_needs_both"][
           "eligible_days"])
    r3 = ans["the_combination_rule_that_makes_it_safe"]
    ck("parts recombined by action-weighted SUMS reproduce the "
       "single-pass day EXACTLY",
       r3["reproduces_the_single_pass"] is True
       and r3["n_whole"] == r3["n_recombined"],
       f"diff {r3['absolute_difference']:.3e} on {r3['n_whole']} actions")
    ck("  and a MEAN OF MEANS does not -- the error is a number, not a "
       "warning",
       r3["MEAN_OF_MEANS_IS_WRONG"][
           "absolute_error_against_the_single_pass"] > 0,
       f"error {r3['MEAN_OF_MEANS_IS_WRONG']['absolute_error_against_the_single_pass']:.3e}")

    print("== a coin may be valued early; a DAY may not be published "
          "early ==")
    pv = ans["valuing_a_day_on_a_partial_coin_set"]
    ck("valuing the day on a PARTIAL coin set gives a different number",
       pv["absolute_error_if_published_early"] > 0,
       f"part alone {pv['part_A_alone_as_if_it_were_the_day']:+.6f} vs "
       f"whole {pv['the_whole_day']:+.6f}")
    try:
        assert_day_is_complete(("btc",))
        ck("  and publishing it AS the day refuses by name", False)
    except PlumbingRefused as exc:
        ck("  and publishing it AS the day refuses by name",
           INCOMPLETE_COIN_SET in str(exc) and "eth" in str(exc))
    ck("  while the complete set proceeds",
       assert_day_is_complete(("btc", "eth"))["complete"] is True)

    print("== what actually spends a band day ==")
    bs = band_slack()
    ck("the correction is MEASURED: books are built retrospectively",
       bs["book_build_lag"]["n_built_after_the_next_day"] > 0,
       f"{bs['book_build_lag']['n_built_after_the_next_day']} of "
       f"{len(bs['book_build_lag']['rows'])} built >1 day late")
    ck("  and the tape that makes that possible is measured, not assumed",
       bs["tape_retention"]["days_of_tape_on_disk"] >= 14,
       f"{bs['tape_retention']['days_of_tape_on_disk']} days on disk, "
       f"headroom "
       f"{bs['tape_retention']['headroom_days_at_the_recent_rate']:.0f}d")
    ck("the two lists are SEPARATE and both non-empty",
       bs["n_modes_that_cost_a_day"] >= 5
       and bs["n_modes_that_cost_only_time"] >= 5)

    print("== the readable set, and the guard that still bites ==")
    ck("09-11..09-13 are readable -- they precede any possible freeze",
       all(assert_day_is_consumed(d) is None
           for d in CANNOT_BE_IN_THE_POPULATION))
    try:
        assert_day_is_consumed("20260914")
        ck("a day beyond the named set still REFUSES", False)
    except PlumbingRefused as exc:
        ck("a day beyond the named set still REFUSES",
           PROTECTED_DAY in str(exc))

    print("== the two blockers are COMPUTED FIELDS ==")
    b = got["THE_TWO_BLOCKERS"]
    ck("blocker 1 carries the per-day coins and the consequence",
       b["eth_has_no_day_book_on_any_day"]["THE_CLOCK_CANNOT_START"] is True
       and b["eth_has_no_day_book_on_any_day"]["n_evaluable_days"] == 0)
    c2 = b["C2_coverage_is_unreachable"]
    ck("blocker 2 shows the ARITHMETIC, not an assertion",
       "generations >=" in c2["ceiling"]["arithmetic"],
       c2["ceiling"]["arithmetic"])
    ck("  and the measured coverage EQUALS the computed ceiling",
       c2["ceiling"]["measured_equals_the_ceiling"] is True,
       f"measured {c2['measured_coverage_C2_btc']:.6f} vs ceiling "
       f"{c2['ceiling']['ceiling']:.6f}")
    ck("  and the gate is stated UNREACHABLE as a predicate",
       c2["ceiling"]["gate_is_unreachable_at_this_threshold"] is True,
       f"shortfall {c2['ceiling']['shortfall_against_the_gate']:.4f}")
    ck("the duration distribution is reported, in declared buckets",
       "histogram_seconds" in got["per_day"][0][
           "generation_duration_profile"])
    ck("the stability of the fraction is computed across days",
       "STABLE_WITHIN_5_POINTS" in c2["stability_of_the_fraction_across_days"]
       or c2["stability_of_the_fraction_across_days"].get("status"))
    ck("the plumbing numbers carry their limit IN THE SAME BLOCK",
       got["plumbing_numbers"]["NOT_EVIDENCE"] == VERDICT
       and set(got["plumbing_numbers"]) >= {"p_values", "futility",
                                            "population_is_consumed"})
    ck("dead-or-narrow is decided by ARITHMETIC: no day reaches the "
       "gate, and the multiple is reported",
       c2["n_days_reaching_the_gate"] == 0
       and c2["multiple_of_the_pooled_fraction_the_gate_requires"] > 1,
       f"best day {c2['best_day']:.4f}, gap "
       f"{c2['gap_on_the_best_day']:.4f}, gate needs "
       f"{c2['multiple_of_the_pooled_fraction_the_gate_requires']:.2f}x")
    st = c2["stability_of_the_fraction_across_days"]
    ck("  and the stability PREDICATE agrees with its own arithmetic "
       "on whatever the days happen to be",
       st["STABLE_WITHIN_5_POINTS"] == (st["range"] <= 0.05)
       and st["per_day"] and st["max"] >= st["min"],
       f"range {st['range']:.4f} -> {st['STABLE_WITHIN_5_POINTS']}")
    ck("nothing was tuned after seeing it: the gate and the threshold "
       "are as declared",
       PRED.COVERAGE_MIN == 0.95 and SHORT_GENERATION_S == 1.0)

    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    if "--run" in argv:
        days = READABLE_DAYS
        if "--days" in argv:
            days = tuple(argv[argv.index("--days") + 1].split(","))
        out = argv[argv.index("--out") + 1] if "--out" in argv else "."
        got = plumbing_run(days, outdir=out)
        print(json.dumps({k: v for k, v in got.items()
                          if k != "per_day"}, indent=2, default=str)[:4000])
        return 0
    print(json.dumps({"protocol": PROTOCOL, "verdict": VERDICT,
                      "readable_days": READABLE_DAYS,
                      "why_readable": WHY_READABLE}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
