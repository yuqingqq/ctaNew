"""Is the book's P&L concentration EVIDENCE, or is it arithmetic?

WHY THIS EXISTS. A reading is in circulation that "the top 1% of 4,315
fills carry 113% of the net, the other 99% sum to -13%", and from it that
`r = adverse/spread` on the body already exceeds any plausible overlay
break-even by a factor of four. The arithmetic is right and this module
reproduces it exactly. TWO THINGS ABOUT IT ARE NOT SETTLED, and both are
measured here rather than argued.

1. IT DEPENDS ON WHICH TAIL YOU TAKE, AND THE TWO DISAGREE VIOLENTLY.
   Ranked by SIGNED value the top 43 are the 43 biggest WINNERS and carry
   ~113% -- the remainder must then be negative, because you removed the
   winners. Ranked by ABSOLUTE value they are the 43 most EXTREME fills,
   winners and losers together, and carry ~10%. Same book, same k, two
   answers three orders apart in implication.

2. SELECTING THE TOP k BY OUTCOME AND EVALUATING THE REMAINDER IS
   CONDITIONING ON THE OUTCOME. For ANY book whose P&L is a sum of noisy
   terms with a small positive mean, removing the k best draws leaves a
   negative remainder -- it is close to automatic. So "the other 99% lose
   money" is only evidence if a book with NO heavy tail would not show
   it. THAT IS A NULL, AND IT IS DECLARED AND RUN HERE (rule 6): a
   matched-moment Gaussian book, which has no tail to speak of.

DECIDES NOTHING (rule 14). Whether the overlay case survives is the
policy layer's.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import statistics
import sys
from pathlib import Path

#: Rule 6: declared before the result, and not below this.
MIN_DRAWS = 200


class TailNullRefused(RuntimeError):
    """Refused rather than reporting a concentration with no null."""


def top_k_share(values: list[float], k: int, *, by: str) -> dict:
    """Sum of the top `k` over the sum of all -- with the RANKING NAMED,
    because the answer depends on it and the name is the difference."""
    if by not in ("signed", "abs"):
        raise TailNullRefused(
            f"REFUSED: ranking {by!r} is neither 'signed' nor 'abs'. An "
            f"unnamed ranking is the defect this function exists to stop")
    if not values:
        raise TailNullRefused("REFUSED: no values")
    if not 1 <= k <= len(values):
        raise TailNullRefused(
            f"REFUSED: k={k} outside 1..{len(values)}")
    key = (lambda v: -v) if by == "signed" else (lambda v: -abs(v))
    order = sorted(values, key=key)
    top, body = order[:k], order[k:]
    net = sum(values)
    return {"ranking": by, "k": k, "n": len(values),
            "top_sum": sum(top), "body_sum": sum(body), "net": net,
            "top_share_of_net": (sum(top) / net if abs(net) > 1e-12
                                 else None),
            "body_sum_is_negative": sum(body) < 0,
            "what_the_ranking_means":
                ("the k biggest WINNERS -- removing them must leave a "
                 "smaller remainder, and a negative one whenever they "
                 "exceed the net" if by == "signed" else
                 "the k most EXTREME fills, winners and losers together")}


def gaussian_null(n: int, k: int, mean: float, sd: float, *,
                  draws: int = MIN_DRAWS, seed: int = 0,
                  by: str = "signed") -> dict:
    """THE DECLARED NULL: a book of `n` iid Gaussian fills with the SAME
    mean and sd, which by construction has no heavy tail.

    If a Gaussian book shows the same concentration, the observed
    concentration is a statement about the MEAN BEING SMALL RELATIVE TO
    THE SPREAD, not about the tail -- and the inference drawn from it
    does not go through."""
    if draws < MIN_DRAWS:
        raise TailNullRefused(
            f"REFUSED: {draws} draws is below the declared minimum "
            f"{MIN_DRAWS} (rule 6). An under-sampled correct null "
            f"flatters as much as a wrong one")
    rng = random.Random(seed)
    out = []
    n_neg = 0
    for _ in range(draws):
        v = [rng.gauss(mean, sd) for _ in range(n)]
        r = top_k_share(v, k, by=by)
        # THE STATISTIC IS ILL-BEHAVED WHERE THE NET CAN GO NEGATIVE: a
        # ratio to a net that changes sign is not a share of anything.
        # Found by this module's own falsifier failing at 33% when I had
        # asserted 90%, which is the instrument telling me the statistic
        # is unstable in that regime rather than my threshold being off.
        # It is COUNTED, never dropped in silence (rule 4).
        if r["net"] <= 0:
            n_neg += 1
            continue
        if r["top_share_of_net"] is not None:
            out.append(r["top_share_of_net"])
    out.sort()
    if not out:
        raise TailNullRefused(
            f"REFUSED: every one of {draws} null books had a "
            f"non-positive net, so the share statistic is undefined "
            f"throughout the null and cannot bound anything")
    q = lambda p: out[min(len(out) - 1, max(0, int(p * len(out))))]
    return {"draws": draws, "n_usable": len(out), "by": by,
            "mean": mean, "sd": sd,
            "null_frac_net_non_positive": n_neg / draws,
            "null_median": statistics.median(out) if out else None,
            "null_p05": q(0.05), "null_p95": q(0.95),
            "null_frac_above_1": sum(1 for x in out if x > 1.0) / len(out),
            "design": (f"{draws} books of {n} iid Gaussian fills with the "
                       f"OBSERVED mean and sd; statistic is the top-{k} "
                       f"share of the net under the '{by}' ranking")}


def analyse(values: list[float], *, top_frac: float = 0.01,
            draws: int = MIN_DRAWS, seed: int = 0) -> dict:
    k = max(1, int(round(top_frac * len(values))))
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    obs = {b: top_k_share(values, k, by=b) for b in ("signed", "abs")}
    nul = {b: gaussian_null(len(values), k, mean, sd, draws=draws,
                            seed=seed, by=b) for b in ("signed", "abs")}
    res = {}
    for b in ("signed", "abs"):
        o = obs[b]["top_share_of_net"]
        res[b] = {
            "observed": obs[b], "null": nul[b],
            "observed_share": o,
            "null_median_share": nul[b]["null_median"],
            "observed_inside_null_90pct": (
                None if o is None
                else nul[b]["null_p05"] <= o <= nul[b]["null_p95"]),
            "gaussian_book_also_exceeds_1": nul[b]["null_frac_above_1"] > 0.5,
        }
    return {
        "n": len(values), "k": k, "mean_cents": mean, "sd_cents": sd,
        "net_cents": sum(values),
        "coefficient_of_variation": (sd / mean if mean else None),
        "by_ranking": res,
        "the_two_rankings_disagree": (
            obs["signed"]["top_share_of_net"] is not None
            and obs["abs"]["top_share_of_net"] is not None
            and abs(obs["signed"]["top_share_of_net"]
                    - obs["abs"]["top_share_of_net"]) > 0.5),
        "decides_nothing": "REPORTED (rule 14).",
    }


def selftest() -> int:
    checks, fails = 0, []

    def ok(c, m):
        nonlocal checks
        checks += 1
        if not c:
            fails.append(m)

    # THE TWO RANKINGS MUST BE ABLE TO DISAGREE -- that is the finding.
    v = [100.0, -99.0] + [1.0] * 8      # net 9.0
    sg = top_k_share(v, 1, by="signed")
    ab = top_k_share(v, 1, by="abs")
    ok(abs(sg["top_sum"] - 100.0) < 1e-9 and sg["body_sum_is_negative"] is True
       and abs(ab["top_sum"] - 100.0) < 1e-9,
       f"the two rankings pick the same element when the biggest winner "
       f"is also the most extreme -- {sg['top_sum']} / {ab['top_sum']}")
    sg2 = top_k_share(v, 2, by="signed")
    ab2 = top_k_share(v, 2, by="abs")
    ok(abs(sg2["top_sum"] - 101.0) < 1e-9
       and abs(ab2["top_sum"] - 1.0) < 1e-9
       and sg2["top_share_of_net"] > 1.0
       and ab2["top_share_of_net"] < 1.0,
       f"FALSIFIER: at k=2 the SIGNED tail is the two biggest winners "
       f"({sg2['top_sum']}) and the ABSOLUTE tail is the winner and the "
       f"loser ({ab2['top_sum']}). Same book, same k, "
       f"{sg2['top_share_of_net']:.2f} vs {ab2['top_share_of_net']:.2f} "
       f"of the net -- which is why an unnamed ranking is not a result")
    # A UNIFORM BOOK: the top k must be about k/n, not a tail.
    u = top_k_share([1.0] * 100, 1, by="signed")
    ok(abs(u["top_share_of_net"] - 0.01) < 1e-9,
       "NEGATIVE CONTROL: in a book with no dispersion the top 1% carry "
       "exactly 1% -- so a large share is dispersion, not a law")
    # THE NULL MUST BE ABLE TO EXCEED 1 -- otherwise it cannot show that
    # the observed concentration is arithmetic rather than evidence.
    # AT THE OBSERVED SHAPE (n=4315, k=43, mean~2c), a GAUSSIAN book --
    # no heavy tail by construction -- crosses a top-1% share of 1.0 on
    # dispersion ALONE. Parameters measured by probe, not guessed.
    hi = gaussian_null(4315, 43, mean=2.0, sd=100.0, draws=MIN_DRAWS)
    lo = gaussian_null(4315, 43, mean=2.0, sd=1.0, draws=MIN_DRAWS)
    ok(hi["null_median"] > 1.0 and lo["null_median"] < 0.05
       and hi["null_frac_above_1"] > 0.4 and lo["null_frac_above_1"] == 0.0,
       f"FALSIFIER BOTH WAYS: at the OBSERVED shape a GAUSSIAN book with "
       f"sd=100 has a MEDIAN top-1% share of {hi['null_median']:.3f} -- "
       f"ABOVE 1, WITH NO HEAVY TAIL -- while the same book at sd=1 "
       f"reads {lo['null_median']:.3f}, essentially k/n. So a share "
       f"above 1 is a statement about DISPERSION, and calling it "
       f"evidence of tail-dependence needs this null to be beaten, not "
       f"assumed")
    ok(hi["null_frac_net_non_positive"] < 0.5
       and lo["null_frac_net_non_positive"] == 0.0,
       f"and the regime where the statistic is UNDEFINED is counted "
       f"rather than dropped: {hi['null_frac_net_non_positive']:.0%} of "
       f"the sd=100 null books had a non-positive net, where a 'share of "
       f"the net' is not a share of anything")
    for bad, why in ((("nope",), "an unnamed ranking"),):
        try:
            top_k_share([1.0], 1, by=bad[0])
            ok(False, f"REFUSAL: must refuse {why}")
        except TailNullRefused:
            ok(True, "")
    try:
        gaussian_null(100, 1, 0.0, 1.0, draws=10)
        ok(False, "REFUSAL: must refuse an under-sampled null")
    except TailNullRefused:
        ok(True, "")
    try:
        top_k_share([], 1, by="signed")
        ok(False, "REFUSAL: must refuse an empty book")
    except TailNullRefused:
        ok(True, "")
    print(json.dumps({"selftest": "PASS" if not fails else "FAIL",
                      "checks": checks, "failures": fails}, indent=1))
    return 0 if not fails else 1


def baseline_fill_values(cache: Path) -> tuple[list, list]:
    """(maker P&L, spread) per fill of the NO-CANCEL book, straight from
    the reference -- which round 58 proved IS the 0-cancel arm's received
    set, by exact equality of the markout totals."""
    import harmful_stateful_policy as HSP
    fr = pickle.loads(cache.read_bytes())["fr"]
    pnl, spr = [], []
    for slug, sides in fr["reference"].items():
        for side in HSP.SIDES:
            sgn = 1.0 if side == HSP.SIDES[0] else -1.0
            for g in sides[side]:
                for t in g.get("tranches", ()):
                    sh, mk = t.get("shares"), t.get("markout_cents_per_share")
                    mid, lvl = t.get("mid_at_fill"), t.get("level")
                    if not sh or mk is None or mid is None or lvl is None:
                        continue
                    pnl.append(float(mk) * sh)
                    spr.append(sgn * (float(mid) - float(lvl)) * 100.0 * sh)
    return pnl, spr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default=str(
        Path(__file__).resolve().parents[2]
        / "data/pm_5min/derived/de_section81_cache_v2_12.pkl"))
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import time
    pnl, spr = baseline_fill_values(Path(a.cache))
    res = analyse(pnl, draws=a.draws)
    k = res["k"]
    # r on the body under BOTH rankings, which is where the readings part.
    rows = sorted(zip(pnl, spr), key=lambda x: -x[0])[k:]
    rows_abs = sorted(zip(pnl, spr), key=lambda x: -abs(x[0]))[k:]
    def _r(rs):
        sp = sum(x[1] for x in rs)
        ad = sum(x[0] for x in rs) - sp
        return {"n": len(rs), "spread_cents": sp, "adverse_cents": ad,
                "pnl_cents": sum(x[0] for x in rs),
                "r_adverse_over_spread": (abs(ad) / sp if sp > 0 else None)}
    res["body_r_by_ranking"] = {"signed": _r(rows), "abs": _r(rows_abs),
                                "whole_book": _r(list(zip(pnl, spr)))}
    res["as_of"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["source_cache"] = str(a.cache)
    txt = json.dumps(res, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
