"""THE RECONCILIATION DA SPECIFIED, AS A CHECK RATHER THAN A SENTENCE.

BE 135. At BE 133 I wrote that the equivalence was "falsifiable one seam
over, where DA specified it". REV 142 drove that and it was FALSE AS A
DESCRIPTION OF THE CODE: `tranches_before_placement_latency` and
`KEPT-VALUE` appear ZERO times in `de_point_estimate_day`,
`de_multiday_gate1_runner`, `de_phase4_diag_runner`, `be_cancel_axis_null`
and `da_gate1_day_verdict`. What existed and fired was the per-generation
partition of COUNTS; the equality of MONEY was checked nowhere. A
specification described as a check is the thing this programme has been
finding all night, and I shipped one.

THIS IS THE CHECK. DA's criterion, unchanged:

    KEPT-VALUE + DROPPED-VALUE == ALL-TRANCHE VALUE
    KEPT-VALUE == the baseline total ALREADY IN THE LEDGER, to the digit
                  (37,315.551431 on 09-03, the point estimate's
                   `zero_cancel_baseline_total_cents`)

AND IT NEEDS NO REPLAY, which is what makes it runnable at all. The
baseline total is a RECORDED FIELD of the point-estimate artifact, so the
check READS it and recomputes only the book side. At BE 133 I said the
reconciliation "needs a replay"; it does not -- it needs the replay's
RECORDED RESULT, which exists.

THE VALUATION IS R-801 AND IS DE's OWN FUNCTION, CALLED:
`de_multiday_gate1_runner.settlement_legs_by_slug` over fills built in the
shape `received_fills` builds them -- `px_cents = level * 100.0`,
`sgn = +1` for BUY_UP -- so there is no second copy of the estimand. The
5-second markout is NOT the valuation and is never read here.

BOTH LEGS TRAVEL SEPARATELY (DA's criterion 4): a dropped tranche moves the
RESIDUAL by 100 c/share independent of its price, so a single total would
hide it.

AND THE NUMBER IT PRODUCES IS AN UPPER BOUND. A dropped tranche is a fill
that arrived BEFORE OUR QUOTE COULD REST; valuing it ASSUMES WE WOULD HAVE
GOT IT, which is a fill-probability assumption, and fills are ENDOGENOUS
(CLAUDE.md reliability rule 1). One day is a POINT ESTIMATE WITH NO
INTERVAL; >= 5 complete UTC days before "the latency effect is real" may be
said.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DROPPED_KEY = "tranches_before_placement_latency"
SIDES = ("BUY_UP", "SELL_UP")


class ReconcileRefused(RuntimeError):
    """A named refusal."""


def _fills_from(ref: dict, key: str) -> list:
    """Tranches in the shape `settlement_legs_by_slug` values.

    `px_cents = level * 100.0` and the BUY side is `HSP.SIDES[0]` --
    both read from `de_phase4_diag_runner.received_fills`, not invented."""
    out = []
    for slug in sorted(ref):
        by_side = ref[slug] or {}
        for side in SIDES:
            for g in (by_side.get(side) or ()):
                for t in (g.get(key) or ()):
                    lvl = t.get("level")
                    if lvl is None:
                        lvl = g.get("level")
                    if lvl is None:
                        raise ReconcileRefused(
                            f"REFUSED -- TRANCHE_HAS_NO_LEVEL at "
                            f"{slug}/{side}/gen {g.get('gen')}: neither the "
                            f"tranche nor its generation carries one, so it "
                            f"cannot be valued and a total that skipped it "
                            f"would be short by an unknown amount.")
                    out.append({"slug": slug, "side": side,
                                "px_cents": float(lvl) * 100.0,
                                "size": float(t.get("shares") or 0.0)})
    return out


def value(ref: dict, key: str, winners: dict) -> dict:
    """R-801 legs for one tranche set -- DE's function, called."""
    import de_multiday_gate1_runner as G
    fills = _fills_from(ref, key)
    legs = G.settlement_legs_by_slug(fills, winners)
    per = legs.get("per_slug") or {}
    return {"n_fills": len(fills),
            "trades_leg_cents": sum(v["trades_leg_cents"] for v in per.values()),
            "residual_leg_cents": sum(v["residual_leg_cents"]
                                      for v in per.values()),
            "total_cents": legs.get("total_cents"),
            "n_slugs": legs.get("n_slugs")}


def _both_sets(ref: dict) -> dict:
    """A reference carrying BOTH sets, or a refusal by name."""
    n_gen = n_with = 0
    for slug in ref:
        for side in SIDES:
            for g in ((ref[slug] or {}).get(side) or ()):
                n_gen += 1
                if DROPPED_KEY in g:
                    n_with += 1
    if n_gen and n_with == 0:
        raise ReconcileRefused(
            f"REFUSED -- DROPPED_TRANCHES_ABSENT_FROM_THE_BOOK: none of the "
            f"{n_gen} generations carries `{DROPPED_KEY}`. This book "
            f"predates BE 133, so its dropped tranches were counted and "
            f"discarded and DROPPED-VALUE cannot be computed from it at "
            f"all. The reconciliation is REFUSED, not reported as zero.")
    if n_with != n_gen:
        raise ReconcileRefused(
            f"REFUSED -- DROPPED_TRANCHES_ON_ONLY_SOME_GENERATIONS: "
            f"{n_with} of {n_gen} carry `{DROPPED_KEY}`. A partial split is "
            f"not a split, and a total over it would be short by the "
            f"generations that have no sibling set.")
    return {"n_generations": n_gen, "n_with_the_dropped_key": n_with}


def reconcile(ref: dict, winners: dict, baseline_total_cents: float,
              *, tol: float = 1e-6) -> dict:
    """DA's two equalities, computed and REFUSED on failure."""
    shape = _both_sets(ref)
    kept = value(ref, "tranches", winners)
    dropped = value(ref, DROPPED_KEY, winners)
    allv = value_all(ref, winners)
    sum_total = kept["total_cents"] + dropped["total_cents"]
    if abs(sum_total - allv["total_cents"]) > tol:
        raise ReconcileRefused(
            f"REFUSED -- LEGS_DO_NOT_CLOSE: KEPT {kept['total_cents']!r} + "
            f"DROPPED {dropped['total_cents']!r} = {sum_total!r} against "
            f"ALL-TRANCHE {allv['total_cents']!r}, a difference of "
            f"{sum_total - allv['total_cents']!r}. The two views must "
            f"partition the valued tranches exactly.")
    if abs(kept["total_cents"] - float(baseline_total_cents)) > tol:
        raise ReconcileRefused(
            f"REFUSED -- KEPT_VALUE_DOES_NOT_MATCH_THE_BASELINE: the book's "
            f"kept tranches value to {kept['total_cents']!r} and the "
            f"ledger's zero-cancel baseline total is "
            f"{float(baseline_total_cents)!r}, a difference of "
            f"{kept['total_cents'] - float(baseline_total_cents)!r}. These "
            f"are the same population valued the same way and they must "
            f"agree to the digit.")
    return {
        "protocol": "BE_PLACEMENT_LATENCY_RECONCILE_V1",
        **shape,
        "KEPT": kept, "DROPPED": dropped, "ALL_TRANCHES": allv,
        "kept_plus_dropped_cents": sum_total,
        "baseline_total_cents_from_the_ledger": float(baseline_total_cents),
        "legs_close": True,
        "kept_equals_the_baseline": True,
        "UPPER_BOUND": (
            "DROPPED is an UPPER BOUND on the latency effect, not the "
            "effect: a dropped tranche arrived BEFORE OUR QUOTE COULD REST, "
            "so valuing it ASSUMES WE WOULD HAVE GOT IT -- a "
            "fill-probability assumption, and fills are ENDOGENOUS "
            "(reliability rule 1)."),
        "HOW_IT_MUST_BE_SAID": (
            "On <day> at L=<L>, tranches arriving before the quote could "
            "rest account for {:.6f} cents of settled value UNDER THE "
            "ASSUMPTION THAT EVERY ONE WOULD HAVE FILLED -- an upper bound "
            "on the latency effect.").format(dropped["total_cents"]),
        "SCOPE": "ONE DAY IS A POINT ESTIMATE WITH NO INTERVAL (rule 8); "
                 ">= 5 complete UTC days before 'the latency effect is "
                 "real' may be said.",
        "the_markout_is_not_the_valuation": "the estimand is R-801 -- "
                                            "trades cash flow plus "
                                            "share-delta x settlement. The "
                                            "5-second markout is never read "
                                            "here.",
        "valuation_owner": "de_multiday_gate1_runner.settlement_legs_by_slug,"
                           " CALLED and not re-implemented",
    }


def value_all(ref: dict, winners: dict) -> dict:
    """Both sets together -- the L=0 valuation."""
    import de_multiday_gate1_runner as G
    fills = _fills_from(ref, "tranches") + _fills_from(ref, DROPPED_KEY)
    legs = G.settlement_legs_by_slug(fills, winners)
    per = legs.get("per_slug") or {}
    return {"n_fills": len(fills),
            "trades_leg_cents": sum(v["trades_leg_cents"] for v in per.values()),
            "residual_leg_cents": sum(v["residual_leg_cents"]
                                      for v in per.values()),
            "total_cents": legs.get("total_cents"),
            "n_slugs": legs.get("n_slugs")}


EXPECTED_CHECKS = 10


def falsify() -> int:
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    def refuses(fn, needle, label):
        nonlocal checks
        checks += 1
        try:
            fn()
            print("FAIL: " + label + " (did not refuse)")
            fails.append(label)
        except ReconcileRefused as e:
            hit = needle in str(e)
            print(("PASS: " if hit else "FAIL: ") + label)
            if not hit:
                fails.append(f"{label} ({e})")

    def ref_of(kept, dropped):
        return {"s1": {"BUY_UP": [{"gen": 0, "t0": 0.0, "t1": 9.0,
                                   "level": 0.5, "tranches": list(kept),
                                   DROPPED_KEY: list(dropped)}],
                       "SELL_UP": []}}
    W = {"s1": {"settle_cents": 100.0, "up_won": True}}
    K = [{"t": 0.4, "shares": 10.0, "level": 0.60}]
    D = [{"t": 0.05, "shares": 4.0, "level": 0.25}]
    ref = ref_of(K, D)
    kv = value(ref, "tranches", W)
    dv = value(ref, DROPPED_KEY, W)
    av = value_all(ref, W)
    ok(abs(kv["total_cents"] + dv["total_cents"] - av["total_cents"]) < 1e-9,
       f"THE PARTITION HOLDS IN MONEY: KEPT {kv['total_cents']:.6f} + "
       f"DROPPED {dv['total_cents']:.6f} = ALL {av['total_cents']:.6f} -- "
       f"and this is the equality that was checked NOWHERE before this "
       f"module existed")
    ok(kv["trades_leg_cents"] != 0 and kv["residual_leg_cents"] != 0
       and dv["residual_leg_cents"] == 4.0 * 100.0,
       f"BOTH LEGS TRAVEL SEPARATELY: kept trades "
       f"{kv['trades_leg_cents']:.2f} / residual "
       f"{kv['residual_leg_cents']:.2f}; dropped residual "
       f"{dv['residual_leg_cents']:.2f} = 4 shares x 100 c -- **the "
       f"residual moves by 100 c/share independent of the price**, which a "
       f"single total would hide")
    r = reconcile(ref, W, kv["total_cents"])
    ok(r["legs_close"] and r["kept_equals_the_baseline"]
       and "UPPER BOUND" in r["UPPER_BOUND"]
       and "NO INTERVAL" in r["SCOPE"]
       and "never read here" in r["the_markout_is_not_the_valuation"],
       "POSITIVE CONTROL: a coherent book reconciles, and the result "
       "carries the upper-bound label, the scope sentence and the "
       "statement that the markout is not the valuation")

    # ---- DRIVEN TO FAIL, which is the whole point (REV 142) -----------
    class _Wrong(dict):
        pass
    import de_multiday_gate1_runner as _G
    _real = _G.settlement_legs_by_slug
    try:
        def _bad(fills, winners):
            out = _real(fills, winners)
            # a DELIBERATELY WRONG dropped value: inflate any valuation of
            # the 4-share tranche by 1 cent
            if any(abs(f["size"] - 4.0) < 1e-9 for f in fills) and \
                    len(fills) == 1:
                out = dict(out, total_cents=out["total_cents"] + 1.0)
            return out
        _G.settlement_legs_by_slug = _bad
        refuses(lambda: reconcile(ref, W, kv["total_cents"]),
                "LEGS_DO_NOT_CLOSE",
                "KNOWN-BAD: a DELIBERATELY WRONG dropped value (one cent "
                "too high) BREAKS the reconciliation by name -- the "
                "falsifier fires, so its passing above is a result and not "
                "a check that has only ever been satisfied")
    finally:
        _G.settlement_legs_by_slug = _real
    refuses(lambda: reconcile(ref, W, kv["total_cents"] + 0.01),
            "KEPT_VALUE_DOES_NOT_MATCH_THE_BASELINE",
            "KNOWN-BAD: a baseline one hundredth of a cent away REFUSES -- "
            "'to the digit' is the test DA specified and it is enforced")
    refuses(lambda: reconcile(
        {"s1": {"BUY_UP": [{"gen": 0, "t0": 0.0, "t1": 9.0, "level": 0.5,
                            "tranches": list(K)}], "SELL_UP": []}},
        W, 0.0), "DROPPED_TRANCHES_ABSENT_FROM_THE_BOOK",
        "KNOWN-BAD, AND IT IS TODAY'S REAL STATE: a book with no "
        "`tranches_before_placement_latency` REFUSES -- every book on disk "
        "predates BE 133, so DROPPED-VALUE cannot be computed from any of "
        "them and this REFUSES rather than reporting zero")
    refuses(lambda: reconcile(
        {"s1": {"BUY_UP": [{"gen": 0, "t0": 0.0, "t1": 9.0, "level": 0.5,
                            "tranches": list(K), DROPPED_KEY: list(D)},
                           {"gen": 1, "t0": 1.0, "t1": 9.0, "level": 0.5,
                            "tranches": []}], "SELL_UP": []}},
        W, 0.0), "DROPPED_TRANCHES_ON_ONLY_SOME_GENERATIONS",
        "KNOWN-BAD: a PARTIAL split REFUSES -- one generation without the "
        "sibling set would make the total short by an unknown amount")
    refuses(lambda: value(
        {"s1": {"BUY_UP": [{"gen": 0, "tranches": [{"t": 1.0,
                                                    "shares": 1.0}]}],
                "SELL_UP": []}}, "tranches", W),
        "TRANCHE_HAS_NO_LEVEL",
        "KNOWN-BAD: a tranche with no level and no generation level "
        "REFUSES rather than being skipped -- a skipped tranche is a total "
        "short by an unknown amount")

    # ---- THE REAL BOOK, WITHOUT TAKING THE LOCK -----------------------
    rc = sorted((HERE.parents[1] / "data" / "pm_5min" / "derived").glob(
        "be_daybook_receipt_20260903_btc__L250ms__EV2*.json"))
    if rc:
        d = json.loads(rc[-1].read_text())
        ok("placement_latency_split" not in d,
           f"AND TODAY'S REAL STATE, FROM THE RECEIPT RATHER THAN A 300 MB "
           f"LOAD: {rc[-1].name} carries NO `placement_latency_split`, so "
           f"the book it names has no dropped set and this reconciliation "
           f"CANNOT RUN on any artifact that exists. It becomes runnable "
           f"with the next build, and until then the honest state is "
           f"REFUSED, not passed")
        pe = sorted((HERE.parents[1] / "data" / "pm_5min" / "derived").glob(
            "p003_de_point_estimate_day_20260903_L250ms__*.json"))
        b = None
        if pe:
            t = json.loads(pe[-1].read_text())
            def find(o):
                if isinstance(o, dict):
                    if "zero_cancel_baseline_total_cents" in o:
                        yield o["zero_cancel_baseline_total_cents"]
                    for v in o.values():
                        yield from find(v)
                elif isinstance(o, list):
                    for v in o:
                        yield from find(v)
            vals = sorted(set(find(t)))
            b = vals[0] if vals else None
        ok(b is not None and abs(b - 37315.551431) < 1e-5,
           f"and the baseline total the check reads is a RECORDED FIELD, "
           f"not a replay: `zero_cancel_baseline_total_cents` = {b!r} in "
           f"{pe[-1].name if pe else 'ABSENT'} -- DA's 37,315.551431. At "
           f"BE 133 I said the reconciliation needed a replay; it needs the "
           f"replay's RECORDED RESULT, which exists")
    else:
        for _ in range(2):
            ok(False, "no EV2x receipt on disk, so today's real state could "
                      "not be driven")

    print()
    if fails:
        print(f"{checks} cells, {len(fails)} failures")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} cells, EXPECTED_CHECKS={EXPECTED_CHECKS}")
        print(f"{checks} cells, 1 failures")
        return 1
    print(f"{checks} cells, 0 failures")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--falsify" in argv or "--selftest" in argv:
        return falsify()
    ap = argparse.ArgumentParser()
    ap.add_argument("--book")
    ap.add_argument("--point-estimate")
    ap.add_argument("--winners", default=None)
    a, _ = ap.parse_known_args(argv[1:])
    if not (a.book and a.point_estimate):
        print("usage: be_placement_latency_reconcile.py --falsify | "
              "--book <book.pkl> --point-estimate <pe.json> "
              "[--winners <winners.json>]")
        return 2
    bk = pickle.loads(Path(a.book).read_bytes())
    pe = json.loads(Path(a.point_estimate).read_text())

    def find(o):
        if isinstance(o, dict):
            if "zero_cancel_baseline_total_cents" in o:
                yield o["zero_cancel_baseline_total_cents"]
            for v in o.values():
                yield from find(v)
        elif isinstance(o, list):
            for v in o:
                yield from find(v)
    vals = sorted(set(find(pe)))
    if not vals:
        raise ReconcileRefused(
            "REFUSED -- BASELINE_TOTAL_NOT_IN_THE_POINT_ESTIMATE: no "
            "`zero_cancel_baseline_total_cents`, so there is nothing to "
            "reconcile the kept tranches against.")
    winners = json.loads(Path(a.winners).read_text()) if a.winners else \
        (bk.get("fr", {}).get("winners") or {})
    print(json.dumps(reconcile(bk["fr"]["reference"], winners, vals[0]),
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
