"""TAIL-RANKING FOR THE CEILING SURVEY. V_oracle ITSELF IS DE'S.

THE CEILING FUNCTION IS NOT DUPLICATED HERE. `de_phase4_diag_runner.
value_ceiling` is the instrument of record (USER ruling, 2026-09-04: DE holds
the reusable `value_ceiling` / `ceiling_capture` functions). This module CALLS
it and REFUSES if it cannot import it, rather than keeping a second copy --
the same BE33-R2 rule that put the provenance census on DA's instrument. Two
implementations that agree tell you nothing and two that disagree tell you
which to trust only after an argument.

WHAT IS HERE IS THE PART DE DOES NOT HAVE: the TAIL-EXCLUDED figures under
BOTH RANKINGS. DE established that the two disagree violently -- on the
measured hour the 43 biggest WINNERS carry 1.13 of net while the 43 most
EXTREME carry 0.10 -- and the reviewer WITHDREW its counterweight because of
it. De-tailing must be SYMMETRIC by |P&L|, and a tail measurement that does
not say WHICH TAIL is not a measurement, so both are emitted side by side.

ON THE CITABLE FORM OF THE ABSENCE, corrected: it is NOT true that no value
ceiling has ever been computed in either programme -- DA found counterexamples
in `live/pm_research/` itself by an AST pass over 189 files. What IS citable:
no ceiling in `live/mm_research/`, none in the registers, and NONE ANYWHERE
FOR THE CANCELLATION-OVERLAY LEVER. This module uses only that form.

AND WHY THE CEILING MATTERS MORE THAN r. `r = adverse/spread` is a ratio of
two SUMS and does not determine whether an overlay can pay: on one book's own
totals, varying only how adverse is DISTRIBUTED, r moves 18.63 -> 18.99% while
the ceiling moves 0.00 -> 22.19%. The question lives in the JOINT DISTRIBUTION
across fills, which a ratio of totals discards.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


class ValueCeilingRefused(RuntimeError):
    """A named refusal."""


def value_ceiling(pnl_per_fill, leg: str = "cancellation_overlay") -> dict:
    """DELEGATES to DE's instrument of record. Refuses rather than falling back.

    A local re-implementation is what BE33-R2 removed from the provenance
    census; the same rule applies here and for the same reason."""
    try:
        import de_phase4_diag_runner as DEV
    except Exception as e:                            # noqa: BLE001
        raise ValueCeilingRefused(
            f"REFUSED: the value-ceiling instrument of record "
            f"(`de_phase4_diag_runner.value_ceiling`) could not be imported "
            f"({type(e).__name__}). This module does not keep a second "
            f"implementation to fall back to.") from None
    v = [float(x) for x in pnl_per_fill]
    if not v:
        raise ValueCeilingRefused(
            "REFUSED: no fills. A ceiling over an empty book is not zero, it "
            "is undefined -- and reporting 0.0 would read as 'no opportunity' "
            "when it means 'no data' (rule 11).")
    out = dict(DEV.value_ceiling(v, leg=leg))
    out["instrument"] = "de_phase4_diag_runner.value_ceiling (INSTRUMENT OF RECORD)"
    total = sum(v)
    out.setdefault("maker_pnl_cents", total)
    out["V_oracle_pct_of_maker_pnl"] = (
        100.0 * out.get("V_oracle_cents", out.get("v_oracle", 0.0)) / total
        if total > 0 else None)
    out["maker_pnl_is_non_positive"] = total <= 0
    return out


def tail_excluded(pnl_per_fill, spread_per_fill, adverse_per_fill,
                  frac: float = 0.01) -> dict:
    """r and V_oracle with the tail removed, under BOTH rankings.

    DE established the two rankings disagree violently: on the measured hour
    the 43 biggest WINNERS carry 1.13 of net while the 43 most EXTREME carry
    0.10, and the reviewer WITHDREW its counterweight because of it.
    De-tailing must be SYMMETRIC by |P&L|; a tail measurement that does not
    say WHICH TAIL is not a measurement, so both are emitted."""
    v = [float(x) for x in pnl_per_fill]
    s = [float(x) for x in spread_per_fill]
    a = [float(x) for x in adverse_per_fill]
    if not (len(v) == len(s) == len(a)):
        raise ValueCeilingRefused(
            f"REFUSED: {len(v)} P&L, {len(s)} spread, {len(a)} adverse. A "
            f"per-fill decomposition must align or the ratio is over "
            f"different fills.")
    n = len(v)
    k = max(1, int(n * frac))
    out = {"n_fills": n, "tail_k": k, "tail_frac": frac}
    for name, order in (("BY_SIGNED_PNL_TOP",
                         sorted(range(n), key=lambda i: -v[i])),
                        ("BY_ABS_PNL_SYMMETRIC",
                         sorted(range(n), key=lambda i: -abs(v[i])))):
        tail = set(order[:k])
        keep = [i for i in range(n) if i not in tail]
        sp_k, ad_k = sum(s[i] for i in keep), sum(a[i] for i in keep)
        out[name] = {
            "r_excluded": (ad_k / sp_k) if sp_k > 0 else None,
            "tail_share_of_spread": (sum(s[i] for i in tail) / sum(s)
                                     if sum(s) else None),
            "tail_share_of_adverse": (sum(a[i] for i in tail) / sum(a)
                                      if sum(a) else None),
            "tail_share_of_pnl": (sum(v[i] for i in tail) / sum(v)
                                  if sum(v) else None),
            "V_oracle_excluded_cents": -sum(v[i] for i in keep if v[i] < 0),
        }
    sp, ad = sum(s), sum(a)
    out["r_raw"] = (ad / sp) if sp > 0 else None
    out["the_two_rankings_disagree"] = (
        out["BY_SIGNED_PNL_TOP"]["tail_share_of_pnl"]
        != out["BY_ABS_PNL_SYMMETRIC"]["tail_share_of_pnl"])
    out["symmetric_is_the_defensible_one"] = (
        "de-tailing by SIGNED top removes only winners and flatters what "
        "remains; |P&L| removes the extremes on both sides, which is what a "
        "tail-sensitivity check is for")
    return out


EXPECTED_CHECKS = 11


def selftest() -> int:
    checks = 0
    fails = []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    c = value_ceiling([10.0, -3.0, 5.0, -2.0])
    ok(abs(c["V_oracle_cents"] - 5.0) < 1e-12,
       f"POSITIVE CONTROL: V_oracle sums |P&L| over LOSING fills only "
       f"({c['V_oracle_cents']}) -- 3 + 2, not the 15 of winners")
    ok(abs(c["oracle_f"] - 0.5) < 1e-12
       and "de_phase4_diag_runner" in c["instrument"],
       f"DELEGATION IS REAL: the numbers come from DE's instrument of record "
       f"({c['instrument'][:46]}…) and the oracle's decline fraction "
       f"{c['oracle_f']:.2f} is reported beside the ceiling")
    ok(abs(c["V_oracle_pct_of_maker_pnl"] - 50.0) < 1e-9,
       f"and as a PERCENTAGE of maker P&L ({c['V_oracle_pct_of_maker_pnl']:.1f}"
       f"%), the only form comparable across cells of different size")
    import de_phase4_diag_runner as _DEV
    ok(value_ceiling([10.0, -3.0, 5.0, -2.0])["V_oracle_cents"]
       == _DEV.value_ceiling([10.0, -3.0, 5.0, -2.0])["V_oracle_cents"],
       "and it is NOT a second implementation: the delegated result EQUALS "
       "DE's directly -- BE33-R2's rule, one instrument of record")
    z = value_ceiling([4.0, 3.0, 2.0])
    ok(z["V_oracle_cents"] == 0.0 and z["oracle_f"] == 0.0,
       "KNOWN-BAD FOR THE OVERLAY CASE: a book with NO losing fill has a "
       "ceiling of EXACTLY ZERO -- no ranker can add a cent, which is the "
       "verdict r cannot deliver")
    try:
        value_ceiling([])
        ok(False, "empty book must refuse")
    except ValueCeilingRefused as e:
        ok("not zero, it is undefined" in str(e),
           "KNOWN-BAD: an EMPTY book REFUSES rather than reporting 0.0 -- a "
           "zero there would read as 'no opportunity' when it means 'no data'")
    n = value_ceiling([-1.0, -2.0])
    ok(n["maker_pnl_is_non_positive"] and n["V_oracle_pct_of_maker_pnl"] is None,
       "a book with NON-POSITIVE maker P&L reports the percentage as None "
       "rather than a negative or infinite ratio")

    # the two tail rankings must be shown to DISAGREE
    v = [100.0, 90.0, -95.0, 1.0, 2.0, -1.0, 3.0, -2.0, 4.0, -3.0]
    s = [5.0] * 10
    a = [1.0] * 10
    t = tail_excluded(v, s, a, frac=0.2)
    ok(t["tail_k"] == 2,
       f"tail is k = max(1, n*frac) = {t['tail_k']} of {t['n_fills']}")
    ok(t["BY_SIGNED_PNL_TOP"]["tail_share_of_pnl"]
       != t["BY_ABS_PNL_SYMMETRIC"]["tail_share_of_pnl"],
       f"THE TWO RANKINGS DISAGREE, DRIVEN: signed-top tail carries "
       f"{t['BY_SIGNED_PNL_TOP']['tail_share_of_pnl']:.3f} of net and "
       f"|P&L|-symmetric carries "
       f"{t['BY_ABS_PNL_SYMMETRIC']['tail_share_of_pnl']:.3f} -- a tail "
       f"measurement that does not say WHICH is not a measurement")
    ok(t["the_two_rankings_disagree"] is True,
       "and the artifact SAYS they disagree, as a computed field")
    try:
        tail_excluded([1.0], [1.0, 2.0], [1.0])
        ok(False, "misaligned decomposition must refuse")
    except ValueCeilingRefused as e:
        ok("must align" in str(e),
           "KNOWN-BAD: a misaligned per-fill decomposition REFUSES -- the "
           "ratio would otherwise be over different fills")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}.")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    print("usage: be_value_ceiling.py --selftest")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
