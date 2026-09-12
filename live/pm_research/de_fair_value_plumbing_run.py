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
import de_fair_value_predictive as PRED            # noqa: E402
import de_fair_value_rehearsal as REH              # noqa: E402

PROTOCOL = "P003_DE_FAIR_VALUE_PLUMBING_RUN_V1"
VERDICT = "PLUMBING_ONLY_NOT_EVIDENCE"

#: Days this run may touch AT ALL: consumed, and named one by one rather
#: than by a range that would quietly grow.
CONSUMED_DAYS = ("20260903", "20260904", "20260905", "20260906",
                 "20260907", "20260908", "20260909", "20260910")
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
    if day not in CONSUMED_DAYS:
        raise PlumbingRefused(
            f"REFUSED {PROTECTED_DAY}: {day} is not in the consumed set "
            f"{CONSUMED_DAYS}. A plumbing test is still a LOOK, and rule "
            f"11 does not care about intent -- 09-11 onward are live and "
            f"BE is mid-decision on the first of them.")


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


def plumbing_run(days=CONSUMED_DAYS, outdir=None, scope_override=None) -> dict:
    """THE WHOLE §8 PATH ON REAL ROWS. Not evidence, and it says so."""
    outdir = Path(outdir) if outdir else Path(".")
    per_day, blind, refusals, statuses = [], [], [], Counter()
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
        row = {"day": day, "book": dd["book"], "coins_with_books": coins,
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
        "eligibility_as_declared": elig, "accrual_as_declared": acc,
        "coverage_gate": coverage,
        "coverage_gate_C2": coverage_c2,
        "coverage_note":
            "C1 covers every Identity-eligible action by construction; C2 "
            "abstains on generations shorter than "
            f"{SHORT_GENERATION_S}s, so its coverage is a MEASURED share "
            "of the real rows",
        "real_exclusion_statuses": dict(statuses),
        "increments": incs, "p_values": pvals, "holm": holm,
        "exact_sign_test": {c: PRED.exact_sign_p(v)
                            for c, v in incs.items()},
        "futility": {c: PRED.futility(v) for c, v in incs.items()},
        "refusals_encountered": refusals,
    }
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
    print("== the protected days are refused BY NAME ==")
    for day in ("20260911", "20260912", "20260913"):
        try:
            assert_day_is_consumed(day)
            ck(f"{day} is refused", False)
        except PlumbingRefused as exc:
            ck(f"{day} is refused", PROTECTED_DAY in str(exc))
    ck("and a consumed day is admitted", assert_day_is_consumed("20260907")
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
    got = plumbing_run(("20260907",), outdir=out)
    r = got["per_day"][0]
    ck("real actions were built and scored",
       r["n_actions"] > 10000, f"{r['n_actions']} actions on 09-07")
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

    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    if "--run" in argv:
        days = CONSUMED_DAYS
        if "--days" in argv:
            days = tuple(argv[argv.index("--days") + 1].split(","))
        out = argv[argv.index("--out") + 1] if "--out" in argv else "."
        got = plumbing_run(days, outdir=out)
        print(json.dumps({k: v for k, v in got.items()
                          if k != "per_day"}, indent=2, default=str)[:4000])
        return 0
    print(json.dumps({"protocol": PROTOCOL, "verdict": VERDICT,
                      "consumed_days": CONSUMED_DAYS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
